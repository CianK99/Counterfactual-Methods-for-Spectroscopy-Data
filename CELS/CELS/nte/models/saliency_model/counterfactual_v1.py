from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tslearn.neighbors import KNeighborsTimeSeries
from torch.utils.data import TensorDataset, DataLoader

from nte.experiment.utils import tv_norm, save_timeseries
from nte.models.saliency_model import Saliency


class CFExplainer(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(CFExplainer, self).__init__(background_data=background_data,
                                            background_label=background_label,
                                            predict_fn=predict_fn,
                                            )
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.conf_threshold = 0.8
        self.eps = None
        self.eps_decay = 0.9991

    def native_guide_retrieval(self, query, target_label, distance, n_neighbors):
        df = pd.DataFrame(self.background_label, columns=['label'])
        knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)
        knn.fit(self.background_data[list(df[df['label'] == target_label].index.values)])

        dist, ind = knn.kneighbors(query.reshape(1, query.shape[0]), return_distance=True)
        return dist, df[df['label'] == target_label].index[ind[0][:]]

    def cf_label_fun(self, data):
        instance = data 
        device = next(self.predict_fn.parameters()).device
        if torch.is_tensor(instance):
            instance_tensor = instance.clone().detach().to(device)
        else:
            instance_tensor = torch.tensor(instance, dtype=torch.float32).to(device)
        output = self.softmax_fn(self.predict_fn(instance_tensor.reshape(1, 1, -1)))
        
        return np.argmax(output.cpu().data.numpy())

    def learn_global_mask(self, X_bg, Y_bg, model, target=None):
        device = torch.device('cuda' if self.use_cuda else 'cpu')
        model.to(device)
        model.eval()

        mask = torch.full((X_bg.shape[-1],), 0.5, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([mask], lr=self.args.lr)
        softmax = torch.nn.Softmax(dim=1)
        
        batch_size = self.args.global_batch 
        dataset = TensorDataset(X_bg, Y_bg)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"Learning global mask with batch size: {batch_size}")
        print(f"Input shape: {X_bg.shape}, Mask shape: {mask.shape}")

        reference = torch.mean(X_bg, dim=0).to(device)
        print(f"Reference shape: {reference.shape}")

        best_loss = float('inf')
        patience_counter = 0
        patience = 50

        for epoch in range(self.args.global_epochs):
            epoch_loss = 0.0
            for batch_idx, (x_batch, y_batch) in enumerate(loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                X_cf = x_batch * (1 - mask.unsqueeze(0)) + reference.unsqueeze(0) * mask.unsqueeze(0)

                logits = model(X_cf.unsqueeze(1))
                probs = softmax(logits)

                # Determine target class
                if target is None:
                    y_target = 1 - y_batch
                else:
                    y_target = torch.full_like(y_batch, target)

                #Classification loss
                loss_pred = torch.nn.functional.cross_entropy(logits, y_target)
                
                #Sparsity loss
                loss_sparsity = torch.mean(torch.abs(mask))
                
                #Distance loss
                loss_dist = torch.mean(torch.sum(torch.abs(x_batch - X_cf), dim=1))
                
                # Combine losses with better weighting
                loss = loss_pred + 0.01 * loss_sparsity + 0.001 * loss_dist

                loss.backward()
                optimizer.step()
                
                # Clamp mask values
                with torch.no_grad():
                    mask.clamp_(0, 1)

                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            
            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} - no improvement in {patience} epochs")
                break
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{self.args.global_epochs}], Avg Loss: {avg_loss:.4f}")
                print(f"  Loss components - Pred: {loss_pred.item():.4f}, Sparsity: {loss_sparsity.item():.4f}, Dist: {loss_dist.item():.4f}")
                print(f"  Mask stats - Min: {mask.min():.3f}, Max: {mask.max():.3f}, Mean: {mask.mean():.3f}")
                print(f"  Non-zero mask elements: {(mask > 0.1).sum().item()}/{mask.shape[0]}")

        print(f"Final mask stats - Min: {mask.min():.3f}, Max: {mask.max():.3f}, Mean: {mask.mean():.3f}")
        return mask.detach()

    def apply_global_mask(self, X, model, mask):
        device = next(model.parameters()).device
        softmax = torch.nn.Softmax(dim=-1)
        m = mask.to(device)

        if hasattr(self, 'background_data') and len(self.background_data) > 0:
            reference = torch.tensor(np.mean(self.background_data, axis=0), 
                                dtype=torch.float32, device=device)
        else:
            reference = torch.zeros(X.shape[-1], dtype=torch.float32, device=device)

        print(f"Applying global mask - Mask shape: {m.shape}, Reference shape: {reference.shape}")

        cf_list, prob_list = [], []
        with torch.no_grad():
            for i, x in enumerate(X):
                x = x.to(device)
                
                x_cf = x * (1.0 - m) + reference * m
                
                model_input = x_cf.unsqueeze(0).unsqueeze(0)
                
                if i == 0:
                    print(f"  Sample {i}: x shape: {x.shape}, x_cf shape: {x_cf.shape}, model input shape: {model_input.shape}")
                
                p = softmax(model(model_input))
                
                cf_list.append(x_cf.cpu().numpy())
                prob_list.append(p.cpu().numpy()[0])
                
        return cf_list, prob_list

    def generate_saliency(self, data, label, **kwargs):
        self.mode = 'Explore'
        query = data.copy()

        device = next(self.predict_fn.parameters()).device

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32).to(device)
        else:
            data = data.to(device)

        top_prediction_class = np.argmax(kwargs['target'].cpu().data.numpy())
        cf_label = 1 - top_prediction_class  # Flip the class for counterfactual


        dis, idx = self.native_guide_retrieval(query, cf_label, "euclidean", 1)
        NUN = self.background_data[idx.item()]

        
        Rt = torch.tensor(NUN, dtype=torch.float32).to(device)
        mask_init = np.random.uniform(size=[data.shape[-1]], low=0, high=1)
        mask = torch.tensor(mask_init, dtype=torch.float32, device=device, requires_grad=True)

        optimizer = torch.optim.Adam([mask], lr=self.args.lr)
        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)


        for i in range(self.args.max_itr):
            best_loss = float('inf')
            counter = 0
            
            perturbated_input = data * (1 - mask) + Rt * mask

            pred_outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape(1, 1, -1).float()))

            l_maximize = 1 - pred_outputs[0][cf_label]
            l_budget_loss = torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
            l_tv_norm_loss = tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
            
            loss = (self.args.l_budget_coeff * l_budget_loss) + \
                (self.args.l_tv_norm_coeff * l_tv_norm_loss) + \
                (self.args.l_max_coeff * l_maximize)

            if best_loss - loss < 0.0001:
                counter += 1
            else:
                counter = 0
                best_loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.args.enable_lr_decay:
                scheduler.step()

            mask.data.clamp_(0, 1)

            if counter >= 100:
                print("Early stopping triggered: total loss did not improve.")
                break
                
        mask_np = mask.cpu().detach().numpy().flatten()
        if mask_np.max() > mask_np.min():
            mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())
        converted_mask = np.where(mask_np > 0.5, 1, 0)

        converted_mask_tensor = torch.tensor(converted_mask, dtype=torch.float32).to(device)
        perturbated_input = data * (1 - converted_mask_tensor) + Rt * converted_mask_tensor

        pred_outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape(1, 1, -1).float()))
        target_prob = float(pred_outputs[0][cf_label].item())

        return (
            np.asarray(converted_mask),
            perturbated_input.detach().cpu().numpy(),
            float(target_prob),
        )



