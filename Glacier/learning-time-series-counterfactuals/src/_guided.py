import warnings

import numpy as np
import tensorflow as tf
from tensorflow import keras

from wildboar.explain import IntervalImportance
from LIMESegment.Utils.explanations import LIMESegment
from helpers_contiguity import keep_top_k_segments
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class ModifiedLatentCF:
    """Explanations by generating a counterfacutal sample in the latent space of
    any autoencoder.

    References
    ----------
    Learning Time Series Counterfactuals via Latent Space Representations,
    Wang, Z., Samsten, I., Mochaourab, R., Papapetrou, P., 2021.
    in: International Conference on Discovery Science, pp. 369–384. https://doi.org/10.1007/978-3-030-88942-5_29
    """

    def __init__(
        self,
        probability=0.5,
        *,
        tolerance=1e-6,
        max_iter=100,
        optimizer=None,
        autoencoder=None,
        pred_margin_weight=1.0,  # weighted_steps_weight = 1 - pred_margin_weight
        step_weights="local",
        random_state=None,
        lam_tv=1e-3, 
        eps_linf=None,
        contiguity="both",
        k_bands=3,
        band_len=48,
        band_min_len=8
    ):
        """
        Parameters
        ----------
        probability : float, optional
            The desired probability assigned by the model

        tolerance : float, optional
            The maximum difference between the desired and assigned probability

        optimizer :
            Optimizer with a defined learning rate

        max_iter : int, optional
            The maximum number of iterations

        autoencoder : int, optional
            The autoencoder for the latent representation

            - if None the sample is generated in the original space
            - if given, the autoencoder is expected to have `k` decoder layer and `k`
              encoding layers.
        """
        self.optimizer_ = (
            tf.optimizers.Adam(learning_rate=1e-4) if optimizer is None else optimizer
        )
        self.mse_loss_ = keras.losses.MeanSquaredError()
        self.probability_ = tf.constant([probability])
        self.tolerance_ = tf.constant(tolerance)
        self.max_iter = max_iter
        self.autoencoder = autoencoder

        # Weights of the different loss components
        self.pred_margin_weight = pred_margin_weight
        self.weighted_steps_weight = 1 - self.pred_margin_weight

        self.step_weights = step_weights
        self.random_state = random_state
        self.lam_tv = lam_tv         # 0 disables TV
        self.eps_linf = eps_linf     # None disables L tether
        self.contiguity = contiguity
        self.k_bands = k_bands
        self.band_len = band_len
        self.band_min_len = band_min_len

    def fit(self, model):
        """Fit a new counterfactual explainer to the model

        Paramaters
        ----------

        model : keras.Model
            The model
        """
        if self.autoencoder:
            (
                encode_input,
                encode_output,
                decode_input,
                decode_output,
            ) = extract_encoder_decoder(self.autoencoder)
            self.decoder_ = keras.Model(inputs=decode_input, outputs=decode_output)
            self.encoder_ = keras.Model(inputs=encode_input, outputs=encode_output)
        else:
            self.decoder_ = None
            self.encoder_ = None
        self.model_ = model
        return self

    def predict(self, x):
        """Compute the difference between the desired and actual probability

        Parameters
        ---------
        x : Variable
            Variable of the sample
        """
        if self.autoencoder is None:
            z = x
        else:
            z = self.decoder_(x)

        return self.model_(z)

    # The "pred_margin_loss" is designed to measure the prediction probability to the desired decision boundary
    def pred_margin_mse(self, prediction):
        return self.mse_loss_(self.probability_, prediction)

    # An auxiliary MAE loss function to measure the proximity with step_weights
    def weighted_mae(self, original_sample, cf_sample, step_weights):
        return tf.math.reduce_mean(
            tf.math.multiply(tf.math.abs(original_sample - cf_sample), step_weights)
        )

    # An auxiliary normalized L2 loss function to measure the proximity with step_weights
    def weighted_normalized_l2(self, original_sample, cf_sample, step_weights):
        var_diff = tf.math.reduce_variance(original_sample - cf_sample)
        var_orig = tf.math.reduce_variance(original_sample)
        var_cf = tf.math.reduce_variance(cf_sample)

        normalized_l2 = 0.5 * var_diff / (var_orig + var_cf)
        return tf.math.reduce_mean(
            tf.math.multiply(
                normalized_l2,
                step_weights,
            )
        )

    def compute_loss(self, original_sample, z_search, step_weights, target_label):
        loss = tf.zeros(shape=())
        decoded = self.decoder_(z_search) if self.autoencoder is not None else z_search
        pred = self.model_(decoded)[:, target_label]

        loss = self.pred_margin_weight * self.pred_margin_mse(pred)

        weighted_steps_loss = self.weighted_mae(
            original_sample=tf.cast(original_sample, tf.float32),
            cf_sample=tf.cast(decoded, tf.float32),
            step_weights=tf.cast(step_weights, tf.float32),
        )
        loss += (1.0 - self.pred_margin_weight) * weighted_steps_loss

        # Hard constraint on non-editable regions
        if self.contiguity in ["both", "local", "global"]:
            # 1 where NOT editable, 0 where editable
            non_editable_mask = tf.cast(step_weights, decoded.dtype)
            forbidden_changes = (decoded - tf.cast(original_sample, decoded.dtype)) * non_editable_mask
            
            forbidden_loss = tf.reduce_mean(tf.square(forbidden_changes))
            loss += 100.0 * forbidden_loss
        
        # smoothness in editable bands
        if self.lam_tv and self.lam_tv > 0:
            editable_mask = 1.0 - tf.cast(step_weights, decoded.dtype)
            delta = (decoded - tf.cast(original_sample, decoded.dtype)) * editable_mask
            loss += self.lam_tv * self._tv1d(delta)

        # L-inf constraint with higher penalty
        if self.eps_linf is not None:
            editable_mask = 1.0 - tf.cast(step_weights, decoded.dtype)
            delta = (decoded - tf.cast(original_sample, decoded.dtype)) * editable_mask
            overflow = tf.nn.relu(tf.abs(delta) - self.eps_linf)
            loss += 1.0 * tf.reduce_mean(overflow)

        return loss, self.pred_margin_mse(pred), weighted_steps_loss

    # TODO: compatible with the counterfactuals of wildboar
    #       i.e., define the desired output target per label
    def transform(self, x, pred_labels):
        """Generate counterfactual explanations

        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """

        result_samples = np.empty(x.shape)
        losses = np.empty(x.shape[0])
        # `weights_all` needed for debugging
        weights_all = np.empty((x.shape[0], 1, x.shape[1], x.shape[2]))

        for i in range(x.shape[0]):
            if i % 25 == 0:
                print(f"{i+1} samples been transformed.")

            # if self.step_weights == "global" OR "uniform"
            if isinstance(self.step_weights, np.ndarray):  #  "global" OR "uniform"
                step_weights = self.step_weights
            elif self.step_weights == "local":
                # ignore warning of matrix multiplication, from LIMESegment: `https://stackoverflow.com/questions/29688168/mean-nanmean-and-warning-mean-of-empty-slice`
                # ignore warning of scipy package warning, from LIMESegment: `https://github.com/paulvangentcom/heartrate_analysis_python/issues/31`
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    warnings.simplefilter("ignore", category=UserWarning)
                    step_weights = get_local_weights(
                        x[i],
                        self.model_,
                        random_state=self.random_state,
                        pred_label=pred_labels[i],
                        k_bands=self.k_bands,
                        min_len=self.band_min_len,
                    )
            else:
                raise NotImplementedError(
                    "step_weights not implemented, please choose 'local', 'global' or 'uniform'."
                )

            # print(step_weights.reshape(-1))
            x_sample, loss = self._transform_sample(
                x[np.newaxis, i], step_weights, pred_labels[i]
            )

            result_samples[i] = x_sample
            losses[i] = loss
            weights_all[i] = step_weights

        print(f"{i+1} samples been transformed, in total.")

        return result_samples, losses, weights_all

    def _transform_sample(self, x, step_weights, pred_label):
        """Generate counterfactual explanations

        x : array-like of shape [n_samples, n_timestep, n_dims]
            The samples
        """
        lr = float(tf.keras.backend.get_value(self.optimizer_.learning_rate))
        optimizer = tf.optimizers.Adam(learning_rate=lr)
        # TODO: check_is_fitted(self)
        if self.autoencoder is not None:
            z = tf.Variable(self.encoder_(x))
        else:
            z = tf.Variable(x, dtype=tf.float32)

        it = 0
        target_label = 1 - pred_label  # for binary classification

        with tf.GradientTape() as tape:
            loss, pred_margin_loss, weighted_steps_loss = self.compute_loss(
                x, z, step_weights, target_label
            )

        if self.autoencoder is not None:
            pred = self.model_(self.decoder_(z))
        else:
            pred = self.model_(z)

        # # uncomment for debug
        # print(
        #     f"current loss: {loss}, pred_margin_loss: {pred_margin_loss}, weighted_steps_loss: {weighted_steps_loss}, pred prob:{pred}, iter: {it}."
        # )

        # TODO: modify the loss to check both validity and proximity; how to design the condition here?
        # while (pred_margin_loss > self.tolerance_ or pred[:, 1] < self.probability_ or weighted_steps_loss > self.step_tolerance_)?
        # loss > tf.multiply(self.tolerance_rate_, loss_original)
        #
        while (
            pred_margin_loss > self.tolerance_
            or pred[:, target_label] < self.probability_
        ) and (it < self.max_iter if self.max_iter else True):
            # Get gradients of loss wrt the sample
            grads = tape.gradient(loss, z)

            # Update the weights of the sample
            optimizer.apply_gradients([(grads, z)])

            if isinstance(step_weights, np.ndarray) and self.autoencoder is None:
                original_cast = tf.cast(x, z.dtype)
                forbidden_mask = tf.constant(step_weights, dtype=z.dtype)
                projected = z * (1.0 - forbidden_mask) + original_cast * forbidden_mask
                z.assign(projected)

            if self.eps_linf is not None:
                decoded_now = self.decoder_(z) if self.autoencoder is not None else z
                gate = 1.0 - tf.cast(step_weights, decoded_now.dtype)      # 1 in bands
                delta = (decoded_now - tf.cast(x, decoded_now.dtype)) * gate
                overflow = tf.nn.relu(tf.abs(delta) - self.eps_linf)
                # small coefficient so it doesn’t fight the main loss
                with tf.GradientTape() as tape:
                    loss += 1e-1 * tf.reduce_mean(overflow)

            with tf.GradientTape() as tape:
                loss, pred_margin_loss, weighted_steps_loss = self.compute_loss(
                    x, z, step_weights, target_label
                )
            it += 1

            if self.autoencoder is not None:
                pred = self.model_(self.decoder_(z))
            else:
                pred = self.model_(z)

        # # uncomment for debug
        # print(
        #     f"current loss: {loss}, pred_margin_loss: {pred_margin_loss}, weighted_steps_loss: {weighted_steps_loss}, pred prob:{pred}, iter: {it}. \n"
        # )

        res = z.numpy() if self.autoencoder is None else self.decoder_(z).numpy()
        return res, float(loss)
    
    def _tv1d(self, x):  # x: (B,T,1)
        return tf.reduce_mean(tf.abs(x[:, 1:, :] - x[:, :-1, :]))


def extract_encoder_decoder(autoencoder):
    """Extract the encoder and decoder from an autoencoder

    autoencoder : keras.Model
        The autoencoder of `k` encoders and `k` decoders
    """
    depth = len(autoencoder.layers) // 2
    encoder = autoencoder.layers[1](autoencoder.input)
    for i in range(2, depth):
        encoder = autoencoder.layers[i](encoder)

    encode_input = keras.Input(shape=encoder.shape[1:])
    decoder = autoencoder.layers[depth](encode_input)
    for i in range(depth + 1, len(autoencoder.layers)):
        decoder = autoencoder.layers[i](decoder)

    return autoencoder.input, encoder, encode_input, decoder


def get_local_weights(input_sample, classifier_model, random_state=None, pred_label=None,
                      k_bands=3, min_len=8):
    n_timesteps, n_dims = input_sample.shape
    desired_label = int(1 - pred_label) if pred_label is not None else 1
    
    print(f"\n=== DEBUG get_local_weights ===")
    print(f"Input shape: {input_sample.shape}")
    print(f"n_timesteps: {n_timesteps}, n_dims: {n_dims}")
    print(f"pred_label: {pred_label}, desired_label: {desired_label}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        seg_imp, seg_idx = LIMESegment(
            input_sample,
            classifier_model,
            model_type=desired_label,
            cp=20,
            window_size=10,
            random_state=random_state,
        )
    
    print(f"LIMESegment output:")
    print(f"  seg_imp shape: {np.array(seg_imp).shape}, values: {seg_imp}")
    print(f"  seg_idx shape: {np.array(seg_idx).shape}, values: {seg_idx}")
    
    band_mask = keep_top_k_segments(
        seg_idx, seg_imp, K=k_bands, min_len=min_len, total_len=n_timesteps, debug=False
    )
    
    print(f"band_mask shape: {band_mask.shape}")
    print(f"band_mask sum: {band_mask.sum()}")
    print(f"band_mask nonzero positions: {np.nonzero(band_mask)[0]}")
    
    weighted_steps = 1.0 - band_mask.astype(float)
    
    print(f"weighted_steps shape: {weighted_steps.shape}")
    print(f"weighted_steps sum: {weighted_steps.sum()}")
    print(f"Editable positions (where weighted_steps=0): {np.nonzero(weighted_steps == 0)[0]}")
    print(f"=== END DEBUG ===\n")
    
    return weighted_steps.reshape(1, n_timesteps, n_dims)


class FittedKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, timesteps, dims):
        self.model = model
        self.timesteps = timesteps
        self.dims = dims

    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = self.timesteps * self.dims
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X3d = X.reshape(-1, self.timesteps, self.dims)
        probs = self.model.predict(X3d, verbose=0)
        return np.argmax(probs, axis=1)

    

def get_global_weights(input_samples, input_labels, classifier_model,
                       random_state=None, k_bands=3, min_len=8, percentile=90):
    n_samples, n_timesteps, n_dims = input_samples.shape
    
    
    clf = FittedKerasClassifier(classifier_model, timesteps=n_timesteps, dims=n_dims)
    clf.fit(input_samples.reshape(n_samples, -1))

    explainer = IntervalImportance(scoring="accuracy", n_intervals=150, random_state=random_state)
    explainer.fit(clf, input_samples.reshape(n_samples, -1), input_labels)

    seg_idx = explainer.components_
    seg_imp = explainer.importances_.mean               # importance per interval
    
    thr = np.percentile(seg_imp, percentile)
    keep = seg_imp >= thr
    seg_idx_f = [seg_idx[i] for i, m in enumerate(keep) if m]
    seg_imp_f = [seg_imp[i] for i, m in enumerate(keep) if m]

    
    if not seg_idx_f:
        seg_idx_f, seg_imp_f = seg_idx, seg_imp

    band_mask = keep_top_k_segments(seg_idx_f, seg_imp_f, K=k_bands, min_len=min_len, 
                                   total_len=n_timesteps, debug=True)
    weighted_steps = 1.0 - band_mask.astype(float)

    
    return weighted_steps.reshape(1, n_timesteps, 1)


"""def get_global_weights(
    input_samples, input_labels, classifier_model, random_state=None
):
    n_samples, n_timesteps, n_dims = input_samples.shape  # n_dims=1

    class ModelWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            p = self.model.predict(X.reshape(n_samples, n_timesteps, 1))
            return np.argmax(p, axis=1)

        def fit(self, X, y):
            return self.model.fit(X, y)

    clf = ModelWrapper(classifier_model)

    i = IntervalImportance(scoring="accuracy", n_intervals=10, random_state=random_state)
    i.fit(clf, input_samples.reshape(input_samples.shape[0], -1), input_labels)

    # calculate the threshold of masking, 75 percentile
    masking_threshold = np.percentile(i.importances_.mean, 75)
    masking_idx = np.where(i.importances_.mean >= masking_threshold)

    weighted_steps = np.ones(n_timesteps)
    seg_idx = i.n_intervals
    for start_idx in masking_idx[0]:
        weighted_steps[seg_idx[start_idx][0] : seg_idx[start_idx][1]] = 0

    # need to reshape for multiplication in `tf.math.multiply()`
    weighted_steps = weighted_steps.reshape(1, n_timesteps, 1)
    return weighted_steps"""
