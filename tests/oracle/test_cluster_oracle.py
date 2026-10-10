import numpy as np
import pytest
import statsmodels.api as sm

from isbetter.methods.cluster import cluster_bootstrap, cluster_robust_diff


def _data(seed, G=25, per=(3, 12), delta=0.3):
    rng = np.random.default_rng(seed)
    ca, cb, va, vb = [], [], [], []
    for g in range(G):
        eff = rng.normal(0, 1)
        for c, v, shift in ((ca, va, delta), (cb, vb, 0.0)):
            m = int(rng.integers(*per))
            c += [f"g{g}"] * m
            v += list(shift + eff + rng.normal(0, 1, m))
    return np.array(va), np.array(ca), np.array(vb), np.array(cb)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cr1_matches_statsmodels(seed):
    va, ca, vb, cb = _data(seed)
    r = cluster_robust_diff(va, ca, vb, cb)
    y = np.r_[va, vb]
    X = sm.add_constant(np.r_[np.ones(len(va)), np.zeros(len(vb))])
    groups = np.r_[ca, cb]
    fit = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}, use_t=True
    )
    assert r.estimate == pytest.approx(fit.params[1], abs=1e-10)
    assert r.diagnostics["se"] == pytest.approx(fit.bse[1], abs=1e-10)
    assert r.p_value == pytest.approx(fit.pvalues[1], abs=1e-10)
    ci = fit.conf_int()[1]
    assert (r.ci_low, r.ci_high) == pytest.approx(tuple(ci), abs=1e-10)
    assert r.df == len(set(groups)) - 1  # t(G-1), as statsmodels uses for p-values/CI


def test_bootstrap_ci_close_to_robust():
    va, ca, vb, cb = _data(5, G=60)
    rob = cluster_robust_diff(va, ca, vb, cb)
    bs = cluster_bootstrap(va, ca, vb, cb, np.random.default_rng(0), n_resamples=3000)
    assert bs.estimate == pytest.approx(rob.estimate, abs=1e-12)
    assert bs.diagnostics["se"] == pytest.approx(rob.diagnostics["se"], rel=0.15)


def test_ignoring_clusters_understates_se():
    from isbetter.methods.parametric import welch_t

    va, ca, vb, cb = _data(7)
    naive_se = welch_t(va, vb).diagnostics["se"]
    robust_se = cluster_robust_diff(va, ca, vb, cb).diagnostics["se"]
    assert robust_se > naive_se
