import numpy as np

from graft_pillar.train_mask_guided import bootstrap_metric_intervals


def test_bootstrap_metric_intervals_are_deterministic_and_bounded():
    y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 8, dtype=int)
    p = np.array([[0.1, 0.2], [0.2, 0.8], [0.8, 0.3], [0.9, 0.9]] * 8)
    first = bootstrap_metric_intervals(y, p, samples=50, seed=7)
    second = bootstrap_metric_intervals(y, p, samples=50, seed=7)
    assert first == second
    for endpoint in ("recurrence", "complication"):
        for metric in ("auc", "pr_auc", "brier"):
            low, high = first[endpoint][metric]
            assert 0.0 <= low <= high <= 1.0
