"""The psychometric core, pure numpy: 2PL likelihood machinery, EAP ability
estimation, and the synthetic-learner generator the recovery harness fits against.
Every function here is deterministic given its inputs - which is what lets statistical
code be TESTED statistically (fit synthetic learners with known a, b and assert the
fit recovers them)."""
import numpy as np
import pandas as pd

GRID = np.linspace(-4, 4, 81)  # ability grid for EAP


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def p_correct(theta, a, b):
    """2PL: P(correct) = sigmoid(a * (theta - b)), vectorised over responses."""
    return sigmoid(a * (theta - b))


def eap_theta(a, b, y):
    """Expected-a-posteriori ability from one learner's (a, b, y) response arrays,
    with a standard-normal prior - works from the very first response."""
    logpost = -0.5 * GRID ** 2
    for aj, bj, yj in zip(a, b, y):
        p = p_correct(GRID, aj, bj)
        logpost = logpost + np.log(np.where(yj == 1, p, 1 - p) + 1e-12)
    w = np.exp(logpost - logpost.max())
    return float((GRID * w).sum() / w.sum())


def simulate(n_learners, n_items, k_per_learner, seed=0):
    """Synthetic learners with KNOWN parameters - the recovery harness's ground truth.
    a ~ lognormal around 1 (discrimination), b ~ N(0,1) (difficulty), theta ~ N(0,1)."""
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0.0, 0.3, n_items)).clip(0.3, 3.0)
    b = rng.normal(0.0, 1.0, n_items).clip(-3, 3)
    theta = rng.normal(0.0, 1.0, n_learners)
    rows = []
    for i in range(n_learners):
        items = rng.choice(n_items, size=k_per_learner, replace=False)
        y = (rng.random(k_per_learner) < p_correct(theta[i], a[items], b[items]))
        for j, yj in zip(items, y):
            rows.append({"user_id": i, "problem_id": int(j), "correct": int(yj),
                         "order_id": len(rows)})
    return pd.DataFrame(rows), a, b


def halves(df: pd.DataFrame):
    """Per learner, time-ordered: first half (ability evidence) / second half (target)."""
    d = df.sort_values(["user_id", "order_id"]).copy()
    rank = d.groupby("user_id").cumcount()
    size = d.groupby("user_id")["order_id"].transform("size")
    return d[rank < size // 2], d[rank >= size // 2]
