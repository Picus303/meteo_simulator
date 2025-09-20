from pathlib import Path
from simulator.config import RunConfig


def test_config_roundtrip(tmp_path: Path):
    cfg_text = """
constants:
    g: 9.81
numerics:
    dt: 300
physics: {}
outputs:
    freq: 10
    """
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg_text, encoding="utf-8")

    cfg = RunConfig.load(cfg_file)
    assert cfg.constants["g"] == 9.81
    assert cfg.numerics["dt"] == 300
    assert cfg.outputs["freq"] == 10