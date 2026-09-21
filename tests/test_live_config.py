"""artifacts/config.json must be reproducible, not a hand-edited constant."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_shipped_config_matches_the_recomputed_table():
    from leads.live_config import build_config
    shipped = json.loads((REPO / "artifacts" / "config.json")
                         .read_text(encoding="utf-8"))
    fresh = build_config()
    for c in (shipped, fresh):
        c["provenance"].pop("real_rows", None)   # differs between checkouts
    assert fresh == shipped


def test_training_frame_reproduces_the_deployed_models_schema():
    """The imputation table is only trustworthy if the frame it was computed
    from is the frame the model was trained on."""
    from leads.live_config import build_config
    cfg = build_config()
    metrics = json.loads((REPO / "artifacts" / "metrics_rf_full.json")
                         .read_text(encoding="utf-8"))["schema"]
    assert cfg["provenance"]["matches_metrics_schema"] is True
    assert cfg["provenance"]["label_distribution"] == metrics["label_distribution"]


def test_live_config_check_mode_passes_on_the_shipped_file():
    from leads.live_config import main
    assert main(["--check"]) == 0


def test_live_config_check_mode_detects_drift(tmp_path):
    from leads.live_config import main
    shipped = json.loads((REPO / "artifacts" / "config.json")
                         .read_text(encoding="utf-8"))
    shipped["imputation"]["medians"]["message_count"] = 999.0
    bad = tmp_path / "config.json"
    bad.write_text(json.dumps(shipped), encoding="utf-8")
    assert main(["--check", "--out", str(bad)]) == 1
