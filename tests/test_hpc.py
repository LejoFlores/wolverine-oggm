"""The batch-experiment runner (wolverine_oggm.hpc): config parsing, member tables, and one member run end
to end on the offline sample glacier, with a hand-made "prepared" setup. Run with  pytest tests/"""
import json
import os
import textwrap

import pandas as pd
import pytest

os.environ.setdefault("WOLVERINE_OGGM_TEST", "1")

from wolverine_oggm import config, hpc   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write(tmp_path, text, name="exp.yml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_example_configs_parse():
    expdir = os.path.join(REPO, "experiments")
    n = {}
    for f in sorted(os.listdir(expdir)):
        if f.endswith(".yml"):
            conf = hpc.load_config(os.path.join(expdir, f))
            n[conf["name"]] = len(hpc.members(conf))
            for ph in hpc.members(conf).phase.unique():
                if not ph.endswith(".csv"):
                    hpc._phase_curve(ph)
    assert n["smoke_test"] == 4
    assert n["h1_h3_phase_warming_25m"] == 75
    assert n["projections_isimip3b_25m"] == 45
    # experiments that share a setup name must agree on it, or their glacier directories would clash
    keys = {}
    for f in os.listdir(expdir):
        if f.endswith(".yml"):
            s = hpc.load_config(os.path.join(expdir, f))["setup"]
            keys.setdefault(s["name"], set()).add(hpc._setup_key(s))
    assert all(len(v) == 1 for v in keys.values()), keys


def test_members_and_ranges(tmp_path):
    conf = hpc.load_config(_write(tmp_path, """
        name: t
        setup: {name: s, dx: 25, dem: dems/my.tif}
        scenario: {type: random, seeds: [1, 2, 3]}
        factors: {phase: [oggm, threshold2], temperature_bias: [0, 1], sliding: {fraction: 0.5, coupling: [0, 1]}}
        """))
    df = hpc.members(conf)
    assert len(df) == 3 * 2 * 2 * 2
    assert set(df.columns) == {"seed", "phase", "temperature_bias", "precip_temp_offset", "sliding_coupling"}
    assert conf["setup"]["dem"] == str(tmp_path / "dems" / "my.tif")
    assert hpc._dem_source(conf["setup"]) == ("USER", conf["setup"]["dem"])
    assert hpc.parse_range("0-2,5,22-", 24) == [0, 1, 2, 5, 22, 23]
    assert hpc.parse_range("all", 3) == [0, 1, 2]
    gcm = hpc.load_config(_write(tmp_path, """
        name: g
        scenario: {type: gcm, gcms: [mri-esm2-0_r1i1p1f1], ssps: [ssp126, ssp585]}
        factors: {phase: [oggm, jennings90]}
        """, "g.yml"))
    assert len(hpc.members(gcm)) == 4
    assert gcm["setup"]["ssps"] == ["ssp126", "ssp585"]
    assert hpc._dem_source(gcm["setup"]) == ("OGGM_DEFAULT", None)


def test_member_run_and_collect(tmp_path, monkeypatch):
    """Prepare the sample glacier by hand as a setup, then run two members and collect them."""
    monkeypatch.setenv("OGGM_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("WOLVERINE_OGGM_WORKDIR", str(tmp_path / "wd"))
    path = _write(tmp_path, """
        name: hef_test
        setup: {name: hef, glaciers: [RGI50-11.00897], climate: CUSTOM, border: 10}
        scenario: {type: random, nyears: 5, seeds: [1]}
        factors: {phase: [oggm, threshold2], temp_std: 2.5}
        """)
    conf = hpc.load_config(path)
    sdir = hpc.setup_dir(conf)
    import importlib
    importlib.reload(config)
    config.init_oggm(working_dir=sdir)
    gdir = config._test_glacier_directory()
    with open(os.path.join(sdir, "glaciers.txt"), "w") as f:
        f.write(gdir.rgi_id + "\n")
    hpc._write_marker(sdir, "fetch", conf["setup"])
    hpc.write_members(conf)
    assert hpc.run(conf, [0, 1], procs=1) == 0
    out = os.path.join(hpc.results_dir(conf), "members")
    info = json.load(open(os.path.join(out, "m0001.json")))
    assert info["phase"] == "threshold2" and info["calibration"][gdir.rgi_id]["melt_f"] > 0
    ens = hpc.collect(conf)
    assert ens.sizes["member"] == 2 and list(ens.phase.values) == ["oggm", "threshold2"]
    s = pd.read_csv(os.path.join(hpc.results_dir(conf), "summary.csv"))
    assert {"volume_km3", "velocity_mean_myr", "phase"} <= set(s.columns)
    # the shared setup directory is untouched by member runs
    assert not any("_m0000" in f for f in os.listdir(gdir.dir))
