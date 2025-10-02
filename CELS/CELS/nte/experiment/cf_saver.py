from pathlib import Path
import numpy as np

def _append_fields(existing, new):
    merged = {}
    for k in existing:
        merged[k] = np.concatenate((existing[k], new[k]))
    return merged

def save_cf_sample(root, dataset, mode, method, fold, *, x0, xcf, y_true, y_pred, y_cf, saliency=None, tag=None):
    root = Path(root)
    file = root / dataset / mode / f"cf_fold{fold}.npz"
    file.parent.mkdir(parents=True, exist_ok=True)

    x0_batch = np.asarray(x0).reshape(1, -1)
    xcf_batch = np.asarray(xcf).reshape(1, -1)
    
    if saliency is None:
        sal_batch = np.zeros_like(x0_batch)
    else:
        sal_batch = np.asarray(saliency).reshape(1, -1)
    
    tag_batch = np.asarray([str(tag) if tag is not None else ""])

    new_data = dict(
        x0=x0_batch, 
        xcf=xcf_batch, 
        y_true=np.asarray([int(y_true)], dtype=np.int64),   # Ground truth label
        y_pred=np.asarray([int(y_pred)], dtype=np.int64),   # Original model prediction  
        y_cf=np.asarray([int(y_cf)], dtype=np.int64),   # CF model prediction
        saliency=sal_batch, 
        tag=tag_batch
    )

    if file.exists():
        with np.load(file, allow_pickle=True) as Z:
            existing_data = {k: Z[k] for k in Z.files}
        merged_data = _append_fields(existing_data, new_data)
        np.savez_compressed(file, **merged_data)
    else:
        np.savez_compressed(file, **new_data)
        
