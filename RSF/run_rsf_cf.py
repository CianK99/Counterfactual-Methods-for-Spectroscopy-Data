import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wildboar.datasets import (
    load_dataset,
    list_repositories,
    install_repository,
)
if "repotwo" not in list_repositories():
    install_repository("http://127.0.0.1:8765/repo2.json", refresh=True)
from wildboar.ensemble import ShapeletForestClassifier
from wildboar.explain.counterfactual import counterfactuals
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def to_label_binary(y):
    vals = np.unique(y)
    if len(vals) != 2:
        raise ValueError(f"Expected binary labels, got {vals}")
    mapping = {vals[0]: 0, vals[1]: 1}
    return np.vectorize(mapping.get)(y), mapping


def build_global_mask_and_prototypes(X_tr, y_tr01, topk_frac=0.1):
    mu0 = X_tr[y_tr01 == 0].mean(axis=0)
    mu1 = X_tr[y_tr01 == 1].mean(axis=0)
    diff = np.abs(mu1 - mu0)
    k = max(1, int(round(len(diff) * topk_frac)))
    mask_idx = np.argsort(diff)[-k:]          # top-k differing indices
    return mask_idx, mu0, mu1


def apply_global_cf(x, desired, mask_idx, mu0, mu1):
    xcf = x.copy()
    proto = mu1 if desired == 1 else mu0
    xcf[mask_idx] = proto[mask_idx]
    return xcf


def run_single_fold(X_tr, X_te, y_tr01, y_te01, args, fold_idx):
    
    # Train RSF
    rsf = ShapeletForestClassifier(
        n_shapelets=args.n_shapelets,
        metric="euclidean",
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    rsf.fit(X_tr, y_tr01)
    te_acc = accuracy_score(y_te01, rsf.predict(X_te))
    
    # Select subset and targets
    test_size = len(y_te01)
    if test_size > 100:
        test_indices = np.arange(test_size)
        _, _, _, rand_test_idx = train_test_split(
            y_te01,
            test_indices,
            test_size=100,
            random_state=args.random_state,
            stratify=y_te01,
        )
    else:
        rand_test_idx = np.arange(test_size)

    X_sel = X_te[rand_test_idx]
    y_sel = y_te01[rand_test_idx]
    y_true_sel = y_te01[rand_test_idx]
    desired = 1 - y_sel
    
    n = len(rand_test_idx)
    
    # Build CFs
    if args.mode == "local":
        # Wildboar's local CFs
        xcf, _, _ = counterfactuals(
            rsf, X_sel, desired, scoring="euclidean", 
            random_state=args.random_state + fold_idx
        )
        cf_type = "local"
    else:
        # Global CFs via class-mean prototype on a fixed top-k mask
        mask_idx, mu0, mu1 = build_global_mask_and_prototypes(
            X_tr, y_tr01, topk_frac=args.global_topk_frac
        )
        xcf = np.vstack([
            apply_global_cf(x, d, mask_idx, mu0, mu1) for x, d in zip(X_sel, desired)
        ])
        cf_type = "global"
    
    # Calculate metrics
    pred_before = rsf.predict(X_sel)
    pred_after = rsf.predict(xcf)
    
    validity = (pred_after == desired).astype(np.float32)
    proximity = np.linalg.norm(xcf - X_sel, axis=1)
    rel_prox = proximity / (np.linalg.norm(X_sel, axis=1) + 1e-8)
    sparsity = (np.mean(np.abs(xcf - X_sel) > args.distance_thr, axis=1)).astype(np.float32)
    
    # Save fold results
    out_dir = Path(args.output_root) / args.dataset / cf_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"cf_fold{fold_idx}.npz"
    
    np.savez_compressed(
        out_npz,
        x0=X_sel.astype(np.float32),
        xcf=xcf.astype(np.float32),
        y_true=y_true_sel.astype(np.int16),     # ground truth
        y_pred=pred_before.astype(np.int16),    # original predictions
        y_cf=pred_after.astype(np.int16),   # CF predictions
        validity=validity.astype(np.float32),
        proximity=proximity.astype(np.float32),
        rel_prox=rel_prox.astype(np.float32),
        sparsity=sparsity.astype(np.float32),
    )
    
    return {
        'fold': fold_idx,
        'test_acc': te_acc,
        'validity_mean': validity.mean(),
        'rel_prox_mean': rel_prox.mean(),
        'sparsity_mean': sparsity.mean(),
        'proximity_mean': proximity.mean(),
        'n_samples': int(n),
        'npz_path': str(out_npz)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Wildboar dataset name (e.g., EcoliVsKpneumoniae)")
    ap.add_argument("--repository", default="wildboar/ucr",
                    help="Repository id (e.g., wildboar/ucr, repoone/ucr, repotwo/ucr)")
    ap.add_argument("--output-root", default="RSF/cf_runs",
                    help="Root dir to save NPZ like Glacier/CELS")
    ap.add_argument("--metrics-csv", default="RSF/rsf_metrics.csv",
                    help="Append aggregate metrics here")
    ap.add_argument("--samples", type=int, default=250, help="# test samples to CF")
    ap.add_argument("--random-state", type=int, default=4)
    ap.add_argument("--n-shapelets", type=int, default=10)
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--distance-thr", type=float, default=0.025,
                    help="Threshold for sparsity (|xcf-x0| > thr)")
    ap.add_argument("--mode", choices=["local", "global"], default="local")
    ap.add_argument("--global-topk-frac", type=float, default=0.1,
                    help="Fraction of time-steps to edit in global mode (e.g., 0.1 = top 10%)")
    ap.add_argument("--n-splits", type=int, default=5,
                    help="Number of CV folds")
    args = ap.parse_args()

    # Load data
    X, y = load_dataset(args.dataset, repository=args.repository)
    
    y_bin, mapping = to_label_binary(y)
    
    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    fold_results = []
    print(f"[RSF] Running {args.n_splits}-fold CV on {args.dataset}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_bin)):
        print(f"\n[RSF] === Fold {fold_idx + 1}/{args.n_splits} ===")
        
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr01, y_te01 = y_bin[train_idx], y_bin[test_idx]
        
        # Run single fold
        fold_result = run_single_fold(X_tr, X_te, y_tr01, y_te01, args, fold_idx)
        fold_results.append(fold_result)
        
        print(f"[RSF] Fold {fold_idx + 1} - Test Acc: {fold_result['test_acc']:.3f}, "
              f"Validity: {fold_result['validity_mean']:.3f}, "
              f"RelProx: {fold_result['rel_prox_mean']:.3f}, "
              f"Sparsity: {fold_result['sparsity_mean']:.3f}")
    
    # Aggregate results across folds
    metrics = ['test_acc', 'validity_mean', 'rel_prox_mean', 'sparsity_mean', 'proximity_mean']
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        aggregated[f"{metric}_mean"] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)
    
    print(f"\n[RSF] === FINAL RESULTS ({args.n_splits}-fold CV) ===")
    print(f"Test Accuracy: {aggregated['test_acc_mean']:.3f} +- {aggregated['test_acc_std']:.3f}")
    print(f"Validity: {aggregated['validity_mean_mean']:.3f} +- {aggregated['validity_mean_std']:.3f}")
    print(f"Relative Prox: {aggregated['rel_prox_mean_mean']:.3f} +- {aggregated['rel_prox_mean_std']:.3f}")
    print(f"Sparsity: {aggregated['sparsity_mean_mean']:.3f} +- {aggregated['sparsity_mean_std']:.3f}")
    
    # Save aggregated results to CSV
    out_csv = Path(args.metrics_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary row with mean +- std
    row = pd.DataFrame([{
        "dataset": args.dataset,
        "repository": args.repository,
        "mode": args.mode,
        "method": "RSF",
        "cv_type": f"{args.n_splits}fold",
        "n_samples_total": sum(r['n_samples'] for r in fold_results),
        "test_acc_mean": aggregated['test_acc_mean'],
        "test_acc_std": aggregated['test_acc_std'],
        "validity_mean": aggregated['validity_mean_mean'],
        "validity_std": aggregated['validity_mean_std'],
        "rel_prox_mean": aggregated['rel_prox_mean_mean'],
        "rel_prox_std": aggregated['rel_prox_mean_std'],
        "sparsity_mean": aggregated['sparsity_mean_mean'],
        "sparsity_std": aggregated['sparsity_mean_std'],
        "proximity_mean": aggregated['proximity_mean_mean'],
        "proximity_std": aggregated['proximity_mean_std'],
        "n_shapelets": args.n_shapelets,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "thr": args.distance_thr,
        "random_state": args.random_state,
        "output_dir": str(Path(args.output_root) / args.dataset / args.mode / "rsf"),
    }])
    
    if out_csv.exists():
        row.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        row.to_csv(out_csv, index=False)
    
    print(f"[RSF] Saved aggregated CV results to {out_csv}")
    
    # Save individual fold results
    fold_csv = out_csv.parent / f"{out_csv.stem}_folds.csv"
    fold_df = pd.DataFrame(fold_results)
    fold_df['dataset'] = args.dataset
    fold_df['repository'] = args.repository
    fold_df['mode'] = args.mode
    fold_df['method'] = 'RSF'
    
    if fold_csv.exists():
        fold_df.to_csv(fold_csv, mode="a", header=False, index=False)
    else:
        fold_df.to_csv(fold_csv, index=False)
    
    print(f"[RSF] Saved individual fold results to {fold_csv}")


if __name__ == "__main__":
    main()