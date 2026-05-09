import numpy as np


def _safe_trimmed_mean(samples, trim_ratio=0.1):
    """
    samples: np.ndarray, shape [B, S, L, C]
        B: batch size
        S: number of generated samples
        L: pred_len
        C: channels

    return:
        np.ndarray, shape [B, L, C]
    """
    if samples.ndim != 4:
        raise ValueError(
            f"Expected samples with shape [B, S, L, C], got {samples.shape}"
        )

    s = samples.shape[1]
    trim_ratio = float(trim_ratio)

    if trim_ratio <= 0:
        return samples.mean(axis=1)

    if trim_ratio >= 0.5:
        raise ValueError("trim_ratio must be smaller than 0.5")

    k = int(s * trim_ratio)

    if k <= 0:
        return samples.mean(axis=1)

    if 2 * k >= s:
        return samples.mean(axis=1)

    sorted_samples = np.sort(samples, axis=1)
    return sorted_samples[:, k:s - k, :, :].mean(axis=1)


def _safe_mom(samples, mom_groups=5, mom_repeats=3, seed=None):
    """
    Median-of-Means aggregation.

    samples: np.ndarray, shape [B, S, L, C]

    Steps:
        1. Randomly shuffle S samples.
        2. Split them into G groups.
        3. Compute group mean in each group.
        4. Take median over group means.
        5. Repeat R times and average the R medians.

    return:
        np.ndarray, shape [B, L, C]

    Compatibility:
        If sample count is too small for MoM, fallback to mean.
    """
    if samples.ndim != 4:
        raise ValueError(
            f"Expected samples with shape [B, S, L, C], got {samples.shape}"
        )

    b, s, l, c = samples.shape

    mom_groups = int(mom_groups)
    mom_repeats = int(mom_repeats)

    if mom_groups <= 1:
        return samples.mean(axis=1)

    if mom_repeats <= 0:
        mom_repeats = 1

    if s < mom_groups:
        return samples.mean(axis=1)

    group_size = s // mom_groups
    usable = group_size * mom_groups

    if group_size <= 0 or usable <= 0:
        return samples.mean(axis=1)

    rng = np.random.default_rng(seed)
    outs = []

    for _ in range(mom_repeats):
        perm = rng.permutation(s)
        cur = samples[:, perm[:usable], :, :]  # [B, usable, L, C]

        cur = cur.reshape(b, mom_groups, group_size, l, c)
        group_means = cur.mean(axis=2)  # [B, G, L, C]

        # Median across group dimension.
        cur_out = np.median(group_means, axis=1)  # [B, L, C]
        outs.append(cur_out)

    return np.stack(outs, axis=0).mean(axis=0)


def aggregate_samples(
    samples,
    method="mean",
    trim_ratio=0.1,
    mom_groups=5,
    mom_repeats=3,
    seed=None,
):
    """
    Aggregate diffusion samples into point forecasts.

    samples:
        np.ndarray, shape [B, S, L, C]

    method:
        mean:
            original TMDM-r behavior.
        median:
            median over generated samples.
        trimmed_mean:
            remove two-side outlier samples then average.
        mom:
            Median-of-Means aggregation.

    return:
        np.ndarray, shape [B, L, C]
    """
    method = str(method).lower()

    if method == "mean":
        return samples.mean(axis=1)

    if method == "median":
        return np.median(samples, axis=1)

    if method == "trimmed_mean":
        return _safe_trimmed_mean(samples, trim_ratio=trim_ratio)

    if method == "mom":
        return _safe_mom(
            samples,
            mom_groups=mom_groups,
            mom_repeats=mom_repeats,
            seed=seed,
        )

    raise ValueError(f"Unknown point aggregation method: {method}")
