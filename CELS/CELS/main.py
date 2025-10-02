import torch
import numpy as np
import os
import json
import wandb
import ssl
import shortuuid
from nte.models.saliency_model.counterfactual_v1 import CFExplainer
import random
from nte.experiment.default_args0 import parse_arguments
import seaborn as sns
from nte.experiment.utils import get_model, dataset_mapper, backgroud_data_configuration, get_run_configuration, number_to_dataset, set_global_seed, _pred_class_1d
from nte.experiment.cf_saver import save_cf_sample
from nte.utils import CustomJsonEncoder
import traceback
from sklearn.model_selection import train_test_split

sns.set_style("darkgrid")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

ENABLE_WANDB = False
WANDB_DRY_RUN = False


BASE_SAVE_DIR = 'results_v1/2312/'
if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"

if __name__ == '__main__':
    from sklearn.model_selection import StratifiedKFold
    
    args = parse_arguments()
    print("Config: \n", json.dumps(args.__dict__, indent=2))

    if args.dataset in number_to_dataset.keys():
        args.dataset = number_to_dataset[args.dataset]
    DATASET_NAME = args.dataset

    PROJECT_NAME = args.pname
    dataset = dataset_mapper(DATASET=args.dataset, args=args)
    
    # Combine data for CV splitting
    X_all = np.concatenate([dataset.train_data, dataset.test_data])
    y_all = np.concatenate([dataset.train_label, dataset.test_label])
    
    # 5-fold CV loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n=== CELS Fold {fold_idx + 1}/5 ===")
        
        # Set fold-specific data
        dataset.train_data = X_all[train_idx]
        dataset.train_label = y_all[train_idx] 
        dataset.test_data = X_all[test_idx]
        dataset.test_label = y_all[test_idx]
        
        # Limit test to 100 samples (match Glacier)
        test_size = len(dataset.test_label)
        if test_size > 100:
            test_indices = np.arange(test_size)
            _, _, _, rand_test_idx = train_test_split(
                dataset.test_label,
                test_indices,
                test_size=100,
                random_state=4,
                stratify=dataset.test_label,
            )
            dataset.test_data = dataset.test_data[rand_test_idx]
            dataset.test_label = dataset.test_label[rand_test_idx]
        
        # Set fold-specific parameters
        args.run_id = fold_idx
        if args.enable_seed:
            set_global_seed(args.seed_value)
        
        TAG = f'{args.algo}-{DATASET_NAME}-{args.background_data}-{args.background_data_perc}-run-{fold_idx}'
        BASE_SAVE_DIR = f'results_v1/2312/{TAG}'
    
        model = get_model(
            dataset=args.dataset,
            input_size=1,
            num_classes=2,
            X_train=dataset.train_data,
            y_train=dataset.train_label
        )
        
        softmax_fn = torch.nn.Softmax(dim=-1)
        bg_data, bg_label, bg_len = backgroud_data_configuration(
            BACKGROUND_DATA=args.background_data,
            BACKGROUND_DATA_PERC=args.background_data_perc,
            dataset=dataset
        )
        print(f"Using {args.background_data_perc}% of background data. Samples: {bg_len}")
        
        config = args.__dict__
        explainer = None
        if args.algo == 'cf':
            explainer = CFExplainer(
                background_data=bg_data[:bg_len], background_label=bg_label[:bg_len],
                predict_fn=model, enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda
            )

        if args.mode == "global":
            print("--- Running in GLOBAL mode ---")
            X_bg = torch.tensor(bg_data[:bg_len], dtype=torch.float32)
            Y_bg = torch.tensor(bg_label[:bg_len], dtype=torch.long)

            target = args.global_target if args.global_target is not None else None
            global_mask = explainer.learn_global_mask(X_bg, Y_bg, model, target=target)
            
            x_test = torch.tensor(dataset.test_data, dtype=torch.float32)
            cf_list, prob_list = explainer.apply_global_mask(x_test, model, global_mask)

            res_path = f"bigdata_cels/{DATASET_NAME}/{args.mode}/"
            os.makedirs(res_path, exist_ok=True)
            cf_res, cf_probs, cf_maps = [], [], []

            for i, (xcf_tensor, prob_tensor) in enumerate(zip(cf_list, prob_list)):
                x0  = dataset.test_data[i].astype(np.float32).reshape(-1)
                xcf = np.asarray(xcf_tensor, dtype=np.float32).reshape(-1)
                sal = np.asarray(global_mask.cpu()).reshape(-1)
                y0  = int(dataset.test_label[i])
                ycf = _pred_class_1d(model, xcf)
                
                save_cf_sample(
                    root="cf_runs", 
                    dataset=DATASET_NAME, 
                    mode=args.mode,     
                    method="cels",
                    fold=args.run_id, 
                    x0=x0, 
                    xcf=xcf, 
                    y_true=int(dataset.test_label[i]),      # Ground truth
                    y_pred=y0,      # Original prediction
                    y_cf=ycf,       # CF prediction
                    saliency=sal,
                    tag=f"{TAG}-global-idx{i}"
                )
                
                cf_res.append(xcf)
                cf_probs.append(prob_tensor)
                cf_maps.append(sal)

            print(f"Saving aggregated results to {res_path}")
            np.save(res_path + 'saliency_cf.npy', np.array(cf_res))
            np.save(res_path + 'saliency_cf_prob.npy', np.array(cf_probs))
            np.save(res_path + 'map_cf.npy', np.array(cf_maps))

        elif args.mode == "local":
            print("--- Running in LOCAL mode ---")
            dataset_len = len(dataset.test_data)
            ds = get_run_configuration(args=args, dataset=dataset, TASK_ID=args.task_id)

            print(f"Dataset length: {dataset_len}")
            print(f"Task ID: {args.task_id}")
            print(f"Samples per task: {args.samples_per_task}")

            ds_list = list(ds)
            print(f"get_run_configuration returned {len(ds_list)} items")
                
        
            ds = iter(ds_list)

            res_path = f"bigdata_cels/{DATASET_NAME}/{args.mode}/"
            os.makedirs(res_path, exist_ok=True)
            cf_res, cf_probs, cf_maps = [], [], []

            for ind, (original_signal, original_label) in ds:
                try:
                    if args.enable_seed_per_instance:
                        set_global_seed(random.randint())
                    metrics = {'epoch': {}}
                    cur_ind = args.single_sample_id if args.run_mode == 'single' else (
                        ind + (int(args.task_id) * args.samples_per_task))
                    UUID = shortuuid.uuid()
                    EXPERIMENT_NAME = f'{args.algo}-{cur_ind}-R{args.run_id}-{UUID}-C{ind}-T{args.task_id}'
                    if cur_ind%50 == 0:
                        print(
                            f" {args.algo}: Working on dataset: {DATASET_NAME} index: {cur_ind} [{((cur_ind + 1) / dataset_len * 100):.2f}% Done]")
                    SAVE_DIR = f'{BASE_SAVE_DIR}/{EXPERIMENT_NAME}'
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    os.makedirs(f"./wandb/{TAG}/", exist_ok=True)
                    config['save_dir'] = SAVE_DIR

                    json.dump(config, open(SAVE_DIR + "/config.json", 'w'), indent=2, cls=CustomJsonEncoder)
                    if ENABLE_WANDB:
                        wandb.init(project=PROJECT_NAME, name=EXPERIMENT_NAME, tags=TAG,
                                config=config, reinit=True, force=True, dir=f"./wandb/{TAG}/")

                    device = next(model.parameters()).device
                    original_signal_tensor = torch.tensor(original_signal, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        if args.bbm == 'dnn':
                            target = softmax_fn(model(original_signal_tensor.reshape(1, 1, -1)))
                        else:
                            raise Exception(f"Black Box model not supported: {args.bbm}")

                    category = np.argmax(target.cpu().data.numpy())
                    if ENABLE_WANDB:
                        wandb.run.summary[f"ori_prediction_class"] = category
                        wandb.run.summary[f"ori_prediction_prob"] = np.max(target.cpu().data.numpy())
                        wandb.run.summary[f"ori_label"] = original_label

                    if args.background_data == "none":
                        explainer.background_data = original_signal_tensor
                        explainer.background_label = original_label

                    converted_mask, perturbation_res, target_prob = explainer.generate_saliency(
                        data=original_signal_tensor.cpu().detach().numpy(), label=original_label,
                        save_dir=SAVE_DIR, target=target, dataset=dataset)


                    cf = perturbation_res.flatten()
                    cf_res.append(cf)


                    x0_tensor  = torch.tensor(original_signal, dtype=torch.float32, device=device).reshape(1, 1, -1)
                    xcf_tensor = torch.tensor(cf, dtype=torch.float32, device=device).reshape(1, 1, -1)

                    y0  = int(original_label)
                    ycf = _pred_class_1d(model, xcf_tensor)

                    x0  = x0_tensor.detach().cpu().numpy().flatten()
                    xcf = xcf_tensor.detach().cpu().numpy().flatten()
                    sal = np.asarray(converted_mask).reshape(-1)


                    save_cf_sample(
                        root="cf_runs",
                        dataset=DATASET_NAME,
                        mode=args.mode,
                        method="cels",
                        fold=args.run_id, 
                        x0=x0, 
                        xcf=xcf, 
                        y_true=int(original_label),     # Ground truth
                        y_pred=y0,      # Original prediction
                        y_cf=ycf,       # CF prediction
                        saliency=sal,
                        tag=f"{TAG}-local-idx{cur_ind}"
                    )

                    
                    pert_res_tensor = torch.tensor(perturbation_res, dtype=torch.float32).to(device)
                    pert_res = softmax_fn(model(pert_res_tensor.reshape(1, 1, -1)))
                    pert_label = np.argmax(pert_res.cpu().data.numpy())

                    cf_probs.append(target_prob)
                    cf_maps.append(converted_mask)

                    if ENABLE_WANDB:
                        wandb.run.summary[f"pert_prediction_class"] = pert_label
                        wandb.run.summary[f"target_prob"] = target_prob
                        wandb.run.summary[f"mask"] = converted_mask
                except Exception as e:
                    err_path = f'/tmp/{TAG}_error.log'
                    with open(err_path, 'a+') as f:
                        f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                        f.write("\n")
                    print(f"[ERROR] {err_path}")
                    raise

            np.save(res_path + 'saliency_cf.npy', np.array(cf_res))
            np.save(res_path + 'saliency_cf_prob.npy', np.array(cf_probs))
            np.save(res_path + 'map_cf.npy', np.array(cf_maps))

        print("Done.")


