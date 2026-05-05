"""
Normalized evaluation functions for trajectory-context consistency.

All metrics are designed to be INVARIANT to:
  - Number of agents (N): use per-agent averages, fractions, or ratios
  - Time window length (T): use per-timestep rates or temporal trend slopes

Trajectories format: np.ndarray of shape (timesteps, agents, 7)
  columns: [x, y, vx, vy, goal_x, goal_y, radius]

Each function returns dimensionless or rate-normalized values so that
a 10-agent / 50-frame scene and a 200-agent / 500-frame scene are
directly comparable.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


# ── helpers ──

def _speeds(trajectories):
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    return np.sqrt(vx**2 + vy**2)


def _active_mask(trajectories):
    return ~np.all(trajectories[:, :, :2] == 0, axis=2)


def _sample_frames(T, max_frames=100):
    """Return evenly-spaced frame indices, capped at max_frames."""
    step = max(1, T // max_frames)
    return np.arange(0, T, step)


def _per_agent_endpoints(trajectories):
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    starts, ends, indices = [], [], []
    for i in range(N):
        m = mask[:, i]
        if m.sum() < 2:
            continue
        frames = np.where(m)[0]
        starts.append(trajectories[frames[0], i, :2])
        ends.append(trajectories[frames[-1], i, :2])
        indices.append(i)
    if not starts:
        return np.empty((0, 2)), np.empty((0, 2)), np.array([], dtype=int)
    return np.array(starts), np.array(ends), np.array(indices)


def _temporal_bins(T, n_bins=5):
    """Split T frames into n_bins roughly-equal temporal segments.
    Returns list of (start, end) index pairs."""
    edges = np.linspace(0, T, n_bins + 1, dtype=int)
    return [(edges[i], edges[i + 1]) for i in range(n_bins) if edges[i] < edges[i + 1]]


def _characteristic_spacing(trajectories):
    """Median nearest-neighbor distance across sampled frames.
    This is the scene's natural inter-agent scale — used to adapt
    all distance thresholds so metrics work regardless of agent count."""
    mask = _active_mask(trajectories)
    T = trajectories.shape[0]
    frames = _sample_frames(T, max_frames=30)
    nn_dists = []
    for t in frames:
        active = mask[t]
        pos = trajectories[t, active, :2]
        if len(pos) < 2:
            continue
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        nn_dists.extend(dists.min(axis=1).tolist())
    if not nn_dists:
        return 1.0  # fallback
    return float(np.median(nn_dists))


def _linear_trend_slope(values):
    """Normalized slope of a linear fit to `values` (list/array).
    Returns slope / mean(|values|) so it is scale-free.
    Positive = increasing trend, negative = decreasing."""
    if len(values) < 2:
        return 0.0
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    # least-squares slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    scale = np.mean(np.abs(y)) + 1e-8
    return float(slope / scale)


# ── speed patterns (already mostly N/T invariant, but add trend) ──

def mean_speed(trajectories):
    """Average speed across all active agent-frames. Already N/T invariant.
    Returns: float (m/s)"""
    spd = _speeds(trajectories)
    mask = _active_mask(trajectories)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(spd[mask]))


def speed_variation_coeff(trajectories):
    """Coefficient of variation of speed (std/mean).
    Dimensionless, invariant to N and T.
    0 = perfectly uniform speed, higher = more heterogeneous."""
    spd = _speeds(trajectories)
    mask = _active_mask(trajectories)
    if mask.sum() == 0:
        return 0.0
    m = float(np.mean(spd[mask]))
    if m < 1e-8:
        return 0.0
    return float(np.std(spd[mask]) / m)


def speed_trend(trajectories, n_bins=5):
    """Temporal trend of mean speed: are agents speeding up or slowing down?
    Returns normalized slope: positive = accelerating, negative = decelerating.
    Invariant to N (per-agent mean per bin) and T (normalized slope)."""
    mask = _active_mask(trajectories)
    spd = _speeds(trajectories)
    T = trajectories.shape[0]
    bins = _temporal_bins(T, n_bins)
    bin_means = []
    for t0, t1 in bins:
        m = mask[t0:t1]
        s = spd[t0:t1]
        if m.sum() == 0:
            bin_means.append(0.0)
        else:
            bin_means.append(float(np.mean(s[m])))
    return _linear_trend_slope(bin_means)


# ── density patterns (normalized per-agent) ──

def mean_local_density(trajectories, radius_mult=1.5):
    """Mean number of neighbors within adaptive radius per agent per frame.
    Radius = radius_mult * characteristic_spacing, so it scales with the
    scene's own inter-agent distance. Returns: float (neighbors/agent).
    Invariant to N (adaptive radius + per-agent) and T (averaged)."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    radius = radius_mult * _characteristic_spacing(trajectories)
    frames = _sample_frames(T)
    neighbor_counts = []
    for t in frames:
        active = mask[t]
        pos = trajectories[t, active, :2]
        if len(pos) < 2:
            continue
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        counts = (dists <= radius).sum(axis=1)
        neighbor_counts.extend(counts.tolist())
    if not neighbor_counts:
        return 0.0
    return float(np.mean(neighbor_counts))


def density_trend(trajectories, radius_mult=1.5, n_bins=5):
    """Temporal trend of per-agent local density (adaptive radius).
    Positive = crowd getting denser per agent, negative = dispersing.
    Invariant to both N (adaptive + per-agent) and T (normalized slope)."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    radius = radius_mult * _characteristic_spacing(trajectories)
    bins = _temporal_bins(T, n_bins)
    bin_densities = []
    for t0, t1 in bins:
        seg_frames = _sample_frames(t1 - t0, max_frames=20) + t0
        counts = []
        for t in seg_frames:
            if t >= T:
                continue
            active = mask[t]
            pos = trajectories[t, active, :2]
            if len(pos) < 2:
                continue
            dists = cdist(pos, pos)
            np.fill_diagonal(dists, np.inf)
            counts.extend((dists <= radius).sum(axis=1).tolist())
        bin_densities.append(float(np.mean(counts)) if counts else 0.0)
    return _linear_trend_slope(bin_densities)


def peak_local_density(trajectories, radius_mult=1.5):
    """Maximum per-agent neighbor count (adaptive radius) at any frame,
    divided by the mean neighbor count → how much denser is the peak
    spot compared to average. Returns: float >= 1.0 (ratio).
    Invariant to N (adaptive radius + ratio) and T (max over samples)."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    radius = radius_mult * _characteristic_spacing(trajectories)
    frames = _sample_frames(T, max_frames=150)
    all_counts = []
    max_count = 0
    for t in frames:
        active = mask[t]
        pos = trajectories[t, active, :2]
        if len(pos) < 2:
            continue
        dists = cdist(pos, pos)
        np.fill_diagonal(dists, np.inf)
        counts = (dists <= radius).sum(axis=1)
        all_counts.extend(counts.tolist())
        max_count = max(max_count, int(counts.max()))
    if not all_counts:
        return 1.0
    mean_count = np.mean(all_counts)
    if mean_count < 1e-8:
        return 1.0
    return float(max_count / mean_count)


# ── spatial patterns (normalized) ──

def spatial_concentration(trajectories, n_grid=5):
    """How non-uniform is the spatial distribution of agents?
    Measures Gini coefficient of agent occupancy across grid cells.
    
    0.0 = perfectly uniform (random walk)
    1.0 = all agents in one cell
    
    SFM agents concentrate near goals/obstacles → high value
    Random agents spread uniformly → low value
    
    This INVERTS the discrimination problem: random scores LOW,
    goal-directed models score HIGH."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    frames = _sample_frames(T, max_frames=50)
    
    all_pos = []
    for t in frames:
        active = mask[t]
        all_pos.append(trajectories[t, active, :2])
    if not all_pos:
        return 0.0
    all_pos = np.concatenate(all_pos, axis=0)
    
    xmin, ymin = all_pos.min(axis=0)
    xmax, ymax = all_pos.max(axis=0)
    
    x_edges = np.linspace(xmin, xmax, n_grid + 1)
    y_edges = np.linspace(ymin, ymax, n_grid + 1)
    
    counts = np.zeros(n_grid * n_grid)
    xi = np.clip(np.searchsorted(x_edges, all_pos[:, 0]) - 1, 0, n_grid - 1)
    yi = np.clip(np.searchsorted(y_edges, all_pos[:, 1]) - 1, 0, n_grid - 1)
    for k in range(len(all_pos)):
        counts[xi[k] * n_grid + yi[k]] += 1
    
    # Gini coefficient
    counts_sorted = np.sort(counts)
    n = len(counts_sorted)
    cumsum = np.cumsum(counts_sorted)
    gini = (2 * np.sum((np.arange(1, n+1) * counts_sorted)) / 
            (n * counts_sorted.sum()) - (n + 1) / n)
    return float(np.clip(gini, 0, 1))


# def neighbor_distance_uniformity(trajectories):
#     """How uniform is the nearest-neighbor distance across agents?
#     Returns coefficient of variation (std/mean) of per-agent NN distances.
#     0 = perfectly evenly spaced, higher = more uneven spacing.
#     Invariant to N and T (dimensionless ratio)."""
#     mask = _active_mask(trajectories)
#     T, N, _ = trajectories.shape
#     frames = _sample_frames(T)
#     nn_dists = []
#     for t in frames:
#         active = mask[t]
#         pos = trajectories[t, active, :2]
#         if len(pos) < 2:
#             continue
#         dists = cdist(pos, pos)
#         np.fill_diagonal(dists, np.inf)
#         nn_dists.extend(dists.min(axis=1).tolist())
#     if not nn_dists:
#         return 0.0
#     arr = np.array(nn_dists)
#     m = arr.mean()
#     if m < 1e-8:
#         return 0.0
#     return float(arr.std() / m)


# def spacing_trend(trajectories, n_bins=5):
#     """Temporal trend of mean nearest-neighbor distance.
#     Positive = agents spreading apart, negative = compacting.
#     Invariant to N and T."""
#     mask = _active_mask(trajectories)
#     T, N, _ = trajectories.shape
#     bins = _temporal_bins(T, n_bins)
#     bin_nns = []
#     for t0, t1 in bins:
#         seg_frames = _sample_frames(t1 - t0, max_frames=20) + t0
#         nns = []
#         for t in seg_frames:
#             if t >= T:
#                 continue
#             active = mask[t]
#             pos = trajectories[t, active, :2]
#             if len(pos) < 2:
#                 continue
#             dists = cdist(pos, pos)
#             np.fill_diagonal(dists, np.inf)
#             nns.append(float(dists.min(axis=1).mean()))
#         bin_nns.append(float(np.mean(nns)) if nns else 0.0)
#     return _linear_trend_slope(bin_nns)


# ── flow / direction patterns ──

def flow_alignment(trajectories):
    """Mean resultant length of velocity unit vectors per frame, averaged.
    0 = chaotic, 1 = perfectly aligned. Already invariant to N and T.
    Returns: float in [0, 1]."""
    mask = _active_mask(trajectories)
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    spd = np.sqrt(vx**2 + vy**2)
    moving = mask & (spd > 0.05)
    if moving.sum() == 0:
        return 0.0
    ux = np.where(moving, vx / np.maximum(spd, 1e-8), 0)
    uy = np.where(moving, vy / np.maximum(spd, 1e-8), 0)
    T = trajectories.shape[0]
    frames = _sample_frames(T)
    consistencies = []
    for t in frames:
        m = moving[t]
        if m.sum() < 2:
            continue
        mean_ux = ux[t, m].mean()
        mean_uy = uy[t, m].mean()
        consistencies.append(np.sqrt(mean_ux**2 + mean_uy**2))
    if not consistencies:
        return 0.0
    return float(np.mean(consistencies))


def flow_alignment_trend(trajectories, n_bins=5):
    """Temporal trend of flow alignment.
    Positive = becoming more organized, negative = becoming more chaotic.
    Invariant to N and T."""
    mask = _active_mask(trajectories)
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    spd = np.sqrt(vx**2 + vy**2)
    moving = mask & (spd > 0.05)
    ux = np.where(moving, vx / np.maximum(spd, 1e-8), 0)
    uy = np.where(moving, vy / np.maximum(spd, 1e-8), 0)
    T = trajectories.shape[0]
    bins = _temporal_bins(T, n_bins)
    bin_alignments = []
    for t0, t1 in bins:
        vals = []
        for t in _sample_frames(t1 - t0, max_frames=20) + t0:
            if t >= T:
                continue
            m = moving[t]
            if m.sum() < 2:
                continue
            mx = ux[t, m].mean()
            my = uy[t, m].mean()
            vals.append(np.sqrt(mx**2 + my**2))
        bin_alignments.append(float(np.mean(vals)) if vals else 0.0)
    return _linear_trend_slope(bin_alignments)


def directional_entropy_normalized(trajectories):
    """Direction entropy normalized to [0, 1] by dividing by max entropy.
    0 = all moving same direction, 1 = uniformly distributed directions.
    Invariant to N (histogram-based) and T (pooled)."""
    mask = _active_mask(trajectories)
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    spd = np.sqrt(vx**2 + vy**2)
    moving = mask & (spd > 0.05)
    if moving.sum() == 0:
        return 0.0
    angles = np.arctan2(vy[moving], vx[moving])
    n_bins = 36
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    max_entropy = np.log2(n_bins)
    return float(entropy / max_entropy)


# ── path patterns ──

def path_linearity(trajectories):
    """Ratio of displacement to total path length per agent, averaged.
    0 = wandering, 1 = straight line. Already N/T invariant.
    Returns: float in [0, 1]."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    ratios = []
    for i in range(N):
        m = mask[:, i]
        if m.sum() < 2:
            continue
        frames = np.where(m)[0]
        pos = trajectories[frames, i, :2]
        displacement = np.linalg.norm(pos[-1] - pos[0])
        diffs = np.diff(pos, axis=0)
        path_len = np.sqrt((diffs**2).sum(axis=1)).sum()
        if path_len < 1e-6:
            continue
        ratios.append(displacement / path_len)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def collision_fraction(trajectories):
    """Per-pair physical-collision rate: dist(i,j) < r_i + r_j.
    Uses the per-agent radius channel (states[..., 6]); matches the
    SFM/PEDSIM convention and TrajNet++/Social-GAN per-pair reporting.
    Returns: float in [0, 1]. Invariant to N and T."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    frames = _sample_frames(T)
    pair_total = 0
    pair_collide = 0
    for t in frames:
        active = mask[t]
        if active.sum() < 2:
            continue
        pos = trajectories[t, active, :2]
        rad = trajectories[t, active, 6]
        dists = cdist(pos, pos)
        rsum = rad[:, None] + rad[None, :]
        iu = np.triu_indices(len(pos), k=1)
        pair_total += len(iu[0])
        pair_collide += int((dists[iu] < rsum[iu]).sum())
    if pair_total == 0:
        return 0.0
    return float(pair_collide / pair_total)


def lingering_fraction(trajectories, speed_threshold=0.3):
    """Fraction of agents with mean speed below threshold.
    Already N/T invariant. Returns: float in [0, 1]."""
    mask = _active_mask(trajectories)
    spd = _speeds(trajectories)
    T, N, _ = trajectories.shape
    n_lingering = 0
    n_active = 0
    for i in range(N):
        m = mask[:, i]
        if m.sum() == 0:
            continue
        n_active += 1
        if spd[m, i].mean() < speed_threshold:
            n_lingering += 1
    if n_active == 0:
        return 0.0
    return float(n_lingering / n_active)


# ── behavioral patterns (trend-focused) ──

def dispersal_score(trajectories):
    """How much agents are dispersing from initial centroid.
    Measures radial alignment of displacement vectors (start→end)
    relative to start centroid. Returns: float in [-1, 1].
    +1 = perfect outward dispersal, -1 = perfect inward convergence.
    Invariant to N (per-agent dot product averaged) and T (endpoint-based)."""
    starts, ends, indices = _per_agent_endpoints(trajectories)
    if len(indices) < 3:
        return 0.0
    centroid = starts.mean(axis=0)
    displacements = ends - starts
    radial_dirs = starts - centroid
    radial_norms = np.linalg.norm(radial_dirs, axis=1, keepdims=True)
    disp_norms = np.linalg.norm(displacements, axis=1, keepdims=True)
    moving = (disp_norms.flatten() > 1e-3) & (radial_norms.flatten() > 1e-3)
    if moving.sum() < 3:
        return 0.0
    radial_unit = radial_dirs[moving] / radial_norms[moving]
    disp_unit = displacements[moving] / disp_norms[moving]
    dots = (radial_unit * disp_unit).sum(axis=1)
    return float(np.mean(dots))


def convergence_score(trajectories):
    """How much agents are converging toward final centroid.
    Returns: float in [-1, 1]. +1 = perfect inward convergence.
    Invariant to N and T."""
    starts, ends, indices = _per_agent_endpoints(trajectories)
    if len(indices) < 3:
        return 0.0
    centroid = ends.mean(axis=0)
    displacements = ends - starts
    toward_centroid = centroid - starts
    tc_norms = np.linalg.norm(toward_centroid, axis=1, keepdims=True)
    disp_norms = np.linalg.norm(displacements, axis=1, keepdims=True)
    moving = (disp_norms.flatten() > 1e-3) & (tc_norms.flatten() > 1e-3)
    if moving.sum() < 3:
        return 0.0
    tc_unit = toward_centroid[moving] / tc_norms[moving]
    disp_unit = displacements[moving] / disp_norms[moving]
    dots = (tc_unit * disp_unit).sum(axis=1)
    return float(np.mean(dots))


def spread_trend(trajectories, n_bins=5):
    """Temporal trend of mean distance from centroid (per-agent).
    Positive = expanding, negative = contracting.
    Invariant to N (per-agent mean) and T (normalized slope)."""
    mask = _active_mask(trajectories)
    T, N, _ = trajectories.shape
    bins = _temporal_bins(T, n_bins)
    bin_spreads = []
    for t0, t1 in bins:
        seg_frames = _sample_frames(t1 - t0, max_frames=20) + t0
        dists_to_centroid = []
        for t in seg_frames:
            if t >= T:
                continue
            active = mask[t]
            pos = trajectories[t, active, :2]
            if len(pos) < 2:
                continue
            centroid = pos.mean(axis=0)
            dists_to_centroid.extend(np.linalg.norm(pos - centroid, axis=1).tolist())
        bin_spreads.append(float(np.mean(dists_to_centroid)) if dists_to_centroid else 0.0)
    return _linear_trend_slope(bin_spreads)

# behaviral trend

def clustering_trend(trajectories, eps_mult=1.5, n_bins=5):
    """Is clustering forming or dissolving over time?
    Positive = clusters forming, negative = clusters breaking up.
    Grounded in: Moussaid et al. (2010) group formation dynamics."""
    T = trajectories.shape[0]
    mask = _active_mask(trajectories)
    eps = eps_mult * _characteristic_spacing(trajectories)
    bins = _temporal_bins(T, n_bins)
    bin_fracs = []
    for t0, t1 in bins:
        positions = []
        for i in range(trajectories.shape[1]):
            m = mask[t0:t1, i]
            if m.sum() == 0:
                continue
            positions.append(trajectories[t0:t1][m, i, :2].mean(axis=0))
        if len(positions) < 3:
            bin_fracs.append(0.0)
            continue
        positions = np.array(positions)
        db = DBSCAN(eps=eps, min_samples=3).fit(positions)
        bin_fracs.append(float((db.labels_ >= 0).sum() / len(db.labels_)))
    return _linear_trend_slope(bin_fracs)


def collision_trend(trajectories, n_bins=5):
    """Are physical collisions increasing or decreasing over time?
    Positive = escalating (panic, crushing), negative = resolving.
    Per-pair physical contact (dist < r_i + r_j); SFM/PEDSIM convention.
    Grounded in: Helbing et al. (2000) escape panic dynamics."""
    mask = _active_mask(trajectories)
    T = trajectories.shape[0]
    bins = _temporal_bins(T, n_bins)
    bin_collisions = []
    for t0, t1 in bins:
        seg_frames = _sample_frames(t1 - t0, max_frames=20) + t0
        pair_total = 0
        pair_collide = 0
        for t in seg_frames:
            if t >= T:
                continue
            active = mask[t]
            if active.sum() < 2:
                continue
            pos = trajectories[t, active, :2]
            rad = trajectories[t, active, 6]
            dists = cdist(pos, pos)
            rsum = rad[:, None] + rad[None, :]
            iu = np.triu_indices(len(pos), k=1)
            pair_total += len(iu[0])
            pair_collide += int((dists[iu] < rsum[iu]).sum())
        bin_collisions.append(pair_collide / pair_total if pair_total else 0.0)
    return _linear_trend_slope(bin_collisions)


def lingering_trend(trajectories, speed_threshold=0.3, n_bins=5):
    """Are more agents stopping over time?
    Positive = crowd freezing/waiting, negative = crowd mobilizing.
    Grounded in: Fruin (1971) pedestrian level of service."""
    mask = _active_mask(trajectories)
    spd = _speeds(trajectories)
    T, N, _ = trajectories.shape
    bins = _temporal_bins(T, n_bins)
    bin_fracs = []
    for t0, t1 in bins:
        m = mask[t0:t1]
        s = spd[t0:t1]
        if m.sum() == 0:
            bin_fracs.append(0.0)
            continue
        per_agent = []
        for i in range(N):
            am = m[:, i]
            if am.sum() == 0:
                continue
            per_agent.append(float(s[am, i].mean() < speed_threshold))
        bin_fracs.append(float(np.mean(per_agent)) if per_agent else 0.0)
    return _linear_trend_slope(bin_fracs)


def entropy_trend(trajectories, n_bins=5):
    """Is movement becoming organized or chaotic over time?
    Positive = becoming more chaotic, negative = becoming organized.
    Grounded in: Helbing et al. (2005) self-organization in crowds."""
    mask = _active_mask(trajectories)
    vx = trajectories[:, :, 2]
    vy = trajectories[:, :, 3]
    spd = np.sqrt(vx**2 + vy**2)
    moving = mask & (spd > 0.05)
    T = trajectories.shape[0]
    bins = _temporal_bins(T, n_bins)
    bin_entropies = []
    for t0, t1 in bins:
        angles = np.arctan2(vy[t0:t1][moving[t0:t1]], 
                           vx[t0:t1][moving[t0:t1]])
        if len(angles) == 0:
            bin_entropies.append(0.0)
            continue
        n_angle_bins = 36
        hist, _ = np.histogram(angles, bins=n_angle_bins, 
                              range=(-np.pi, np.pi))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist)))
        max_entropy = np.log2(n_angle_bins)
        bin_entropies.append(entropy / max_entropy)
    return _linear_trend_slope(bin_entropies)

# ── registry ──

FUNCTION_REGISTRY = {
    "mean_speed": mean_speed,
    "speed_variation_coeff": speed_variation_coeff,
    "speed_trend": speed_trend,
    "mean_local_density": mean_local_density,
    "density_trend": density_trend,
    "peak_local_density": peak_local_density,
    "spatial_concentration": spatial_concentration,
    # "neighbor_distance_uniformity": neighbor_distance_uniformity,
    # "spacing_trend": spacing_trend,
    "flow_alignment": flow_alignment,
    "flow_alignment_trend": flow_alignment_trend,
    "directional_entropy_normalized": directional_entropy_normalized,
    "path_linearity": path_linearity,
    "collision_fraction": collision_fraction,
    "lingering_fraction": lingering_fraction,
    "dispersal_score": dispersal_score,
    "convergence_score": convergence_score,
    "spread_trend": spread_trend,
    "clustering_trend": clustering_trend,
    "collision_trend": collision_trend,
    "lingering_trend": lingering_trend,
    "entropy_trend": entropy_trend,
}
