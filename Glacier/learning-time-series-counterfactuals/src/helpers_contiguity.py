import numpy as np

def _to_pairs(seg_idx, total_len=None):
    seg_idx = np.asarray(seg_idx)
    if seg_idx.ndim == 2 and seg_idx.shape[1] == 2:
        pairs = [(int(s), int(e)) for s, e in seg_idx]
    elif seg_idx.ndim == 1 and seg_idx.size >= 2:
        b = seg_idx.astype(int)
        # ensure sorted and unique
        b = np.unique(np.clip(b, 0, None))
        if total_len is not None:
            # append closing boundary to cover tail if missing
            if b[-1] < total_len:
                b = np.append(b, total_len)
            # ensure start at 0
            if b[0] > 0:
                b = np.insert(b, 0, 0)
        pairs = [(int(b[i]), int(b[i+1])) for i in range(len(b)-1)]
    else:
        pairs = []

    if total_len is not None and pairs:
        # clip pairs into [0, total_len]
        clipped = []
        for s, e in pairs:
            s = max(0, min(int(s), int(total_len)))
            e = max(0, min(int(e), int(total_len)))
            if e > s:
                clipped.append((s, e))
        pairs = clipped
    
    return pairs

def keep_top_k_segments(seg_idx, seg_imp, K=3, min_len=8, total_len=None):
    pairs = _to_pairs(seg_idx, total_len=total_len)
    seg_imp = np.asarray(seg_imp, float)

    if not pairs or seg_imp.size == 0:
        L = int(total_len) if total_len is not None else 0
        return np.zeros(L, float)

    S = min(len(pairs), seg_imp.size)
    pairs = pairs[:S]
    seg_imp = seg_imp[:S]

    T = int(total_len) if total_len is not None else max(e for _, e in pairs)
    
    # Different scoring options
    score_length_weighted = np.asarray([(e - s) * imp for (s, e), imp in zip(pairs, seg_imp)], float)
    score_importance_only = seg_imp.copy()
    score_balanced = np.asarray([np.sqrt(e - s) * imp for (s, e), imp in zip(pairs, seg_imp)], float)
    
    # Use importance-only scoring by default
    score = score_importance_only

    order = np.argsort(score)[::-1]
    mask = np.zeros(T, float)
    kept = 0
    
    for j in order:
        s, e = pairs[j]
        seg_length = e - s
        
        if seg_length < min_len:
            continue
            
        mask[s:e] = 1.0
        kept += 1
            
        if kept >= K:
            break
    
    return mask