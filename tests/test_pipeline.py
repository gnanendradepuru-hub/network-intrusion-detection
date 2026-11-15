"""Smoke tests: preprocessing schema is stable and the API scores flows."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from data import BASE_FEATURES, FlowPreprocessor, load_raw  # noqa: E402


def test_preprocessor_schema_is_frozen():
    train = load_raw("KDDTrain+.txt")
    test = load_raw("KDDTest+.txt")
    pre = FlowPreprocessor().fit(train)
    assert list(pre.transform(test).columns) == list(pre.transform(train.head(5)).columns)


def test_api_scores_and_validates():
    from fastapi.testclient import TestClient
    from app.main import REPLAY, app

    client = TestClient(app)
    flows = [{k: r[k] for k in BASE_FEATURES} for r in REPLAY[:5]]
    r = client.post("/api/score", json={"flows": flows})
    assert r.status_code == 200
    res = r.json()["results"]
    assert len(res) == 5 and all(0 <= x["attack_score"] <= 1 for x in res)

    bad = client.post("/api/score", json={"flows": [{"duration": 1}]})
    assert bad.status_code == 422
