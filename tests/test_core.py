"""Fast checks on the offline sample glacier: run with  WOLVERINE_OGGM_TEST=1 pytest tests/"""
import os

import numpy as np
import pytest

os.environ.setdefault("WOLVERINE_OGGM_TEST", "1")

from wolverine_oggm import config, dynamics, massbalance, phase, schnaps   # noqa: E402


@pytest.fixture(scope="module")
def gdir(tmp_path_factory):
    os.environ["WOLVERINE_OGGM_WORKDIR"] = str(tmp_path_factory.mktemp("wd"))
    import importlib
    importlib.reload(config)
    return config.get_glacier_directory(reset=True)


def test_phase_curves():
    r = phase.Ramp(0, 2)
    assert r(-1) == 1 and r(3) == 0 and np.isclose(r(1), 0.5)
    assert np.allclose(phase.expected_solid_fraction(r, np.array([-1.0, 1.0, 3.0]), 0), [1, 0.5, 0])
    # a spread smooths a threshold into a normal CDF
    f = phase.expected_solid_fraction(phase.Threshold(0.0), 0.0, 2.0)
    assert abs(f - 0.5) < 0.02
    lg = phase.Logistic(1.0, 0.5)
    assert np.isclose(lg.t50(), 1.0, atol=0.01)


def test_fit_logistic_recovers_curve():
    rng = np.random.default_rng(0)
    t = rng.uniform(-5, 5, 5000)
    y = np.clip(phase.Logistic(0.7, 0.6)(t) + rng.normal(0, 0.05, t.size), 0, 1)
    fit = phase.fit_logistic(t, y)
    assert abs(fit.t_half - 0.7) < 0.05 and abs(fit.width - 0.6) < 0.05


def test_default_reproduces_oggm(gdir):
    from oggm.core.massbalance import MonthlyTIModel
    h, _ = massbalance.flowline_heights_and_weights(gdir)
    a, b = MonthlyTIModel(gdir), massbalance.phase_model_class()(gdir)
    for y in (1991, 2000):
        assert np.allclose(a.get_annual_mb(h, y), b.get_annual_mb(h, y))
        assert np.allclose(a.get_monthly_mb(h, y + 0.5), b.get_monthly_mb(h, y + 0.5))


def test_recalibration_matches_reference(gdir):
    cls = massbalance.recalibrate(gdir, massbalance.phase_model_class(phase.Threshold(0.0), temp_std=2.0), "_t")
    y0, y1 = config.calibration_years()
    mb = massbalance.specific_mb_by_scheme(gdir, {"t": cls}, range(y0, y1 + 1)).mean()["t"]
    assert abs(mb - gdir.read_json("mb_calib")["reference_mb"]) < 1


def test_hydro_sliding_runs(gdir):
    inv = dynamics.invert_with_sliding(gdir, sliding_frac=0.3, filesuffix="_fst")
    assert abs(inv["sliding_fraction"] - 0.3) < 0.03
    assert abs(inv["volume_km3"] / inv["volume_km3_no_sliding"] - 1) < 0.01
    from oggm import tasks
    m = tasks.run_random_climate(gdir, nyears=3, y0=1996, halfsize=5, seed=0, output_filesuffix="_hs",
                                 init_model_fls=gdir.read_pickle("model_flowlines", filesuffix="_fst"),
                                 evolution_model=dynamics.HydroSliding, glen_a=inv["glen_a"], fs=inv["fs"],
                                 sliding_coupling=1.0, liquid_ref=300.0)
    fs = np.array([f for _, _, f in m.fs_history])
    assert len(fs) >= 30 and fs.max() > inv["fs"] > fs.min()


def test_schnaps_synthetic_roundtrip(gdir, tmp_path):
    d = schnaps.make_synthetic_case(str(tmp_path / "case"), gdir, start="1995-06-01", hours=24 * 30)
    ds = schnaps.load_case(d, "d03")
    mask = schnaps.glacier_mask(ds, gdir=gdir)
    _, fit = schnaps.phase_curve_from_case(schnaps.phase_samples(ds, mask))
    assert abs(fit.t_half - 0.8) < 0.1 and abs(fit.width - 0.6) < 0.1
    mon = schnaps.monthly_climate(ds, mask)
    comp = schnaps.write_oggm_climate(gdir, mon, filesuffix="_s", min_coverage=0.9)
    assert len(comp) == 1   # June 1995 only (the first hour is dropped, July is partial)
