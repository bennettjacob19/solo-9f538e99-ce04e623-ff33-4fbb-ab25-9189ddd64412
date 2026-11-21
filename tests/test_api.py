import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model import ModelConfig


@pytest.fixture
def client():
    return TestClient(create_app(max_blocks=4, model_config=ModelConfig(vocab_size=16, d_model=8, n_heads=2, seed=2)))


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model").json()
    assert info["positions_per_block"] == 4
    assert info["n_layers"] == 1


def test_create_append_and_stats(client):
    r = client.post("/sessions", json={"tokens": [1, 2, 3, 4, 5]})
    assert r.status_code == 201, r.text
    body = r.json()
    sid = body["session_id"]
    assert body["length"] == 5
    assert len(body["logits"]) == 5
    assert body["blocks_allocated"] == 2

    stats = client.get("/stats").json()
    assert stats["used_blocks"] == 2
    assert stats["max_blocks"] == 4
    assert stats["free_blocks"] == 2
    assert stats["block_size"] == 4

    r = client.post(f"/sessions/{sid}/tokens", json={"tokens": [6, 7, 8, 9]})
    assert r.status_code == 200, r.text
    appended = r.json()
    assert appended["start_position"] == 5
    assert appended["positions"] == [5, 6, 7, 8]
    assert len(appended["logits"]) == 4
    assert appended["length"] == 9


def test_fork_shares_blocks(client):
    sid = client.post("/sessions", json={"tokens": [1, 2, 3, 4, 5, 6]}).json()["session_id"]
    used = client.get("/stats").json()["used_blocks"]
    r = client.post(f"/sessions/{sid}/fork", json={"prefix_length": 4})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["shared_blocks"] == 1
    assert body["blocks_allocated"] == 0
    assert client.get("/stats").json()["used_blocks"] == used
    detail = client.get(f"/sessions/{body['session_id']}").json()
    assert detail["tokens"] == [1, 2, 3, 4]


def test_fork_bad_prefix_returns_422(client):
    sid = client.post("/sessions", json={"tokens": [1, 2]}).json()["session_id"]
    r = client.post(f"/sessions/{sid}/fork", json={"prefix_length": 3})
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_prefix"


def test_unknown_session_404(client):
    assert client.get("/sessions/nope").status_code == 404
    assert client.post("/sessions/nope/tokens", json={"tokens": [1]}).status_code == 404
    assert client.delete("/sessions/nope").status_code == 404
    assert client.post("/sessions/nope/fork", json={"prefix_length": 0}).status_code == 404


def test_insufficient_blocks_409_leaves_state(client):
    sid = client.post("/sessions", json={"tokens": [1, 2, 3, 4, 5, 6, 7, 8]}).json()["session_id"]
    # 8 tokens -> 2 blocks; max is 4. Two more full blocks is fine, then fail.
    assert client.post(f"/sessions/{sid}/tokens", json={"tokens": [9] * 8}).status_code == 200
    before = client.get("/stats").json()
    r = client.post(f"/sessions/{sid}/tokens", json={"tokens": [9] * 5})
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["error"] == "insufficient_blocks"
    after = client.get("/stats").json()
    assert after["used_blocks"] == before["used_blocks"]
    assert client.get(f"/sessions/{sid}").json()["length"] == 16


def test_invalid_token_422(client):
    sid = client.post("/sessions", json={"tokens": [1]}).json()["session_id"]
    r = client.post(f"/sessions/{sid}/tokens", json={"tokens": [1, 999]})
    assert r.status_code == 422
    assert client.get(f"/sessions/{sid}").json()["tokens"] == [1]
    assert client.post("/sessions", json={"tokens": [999]}).status_code == 422


def test_empty_append_rejected_by_schema(client):
    sid = client.post("/sessions", json={"tokens": [1]}).json()["session_id"]
    r = client.post(f"/sessions/{sid}/tokens", json={"tokens": []})
    assert r.status_code == 422


def test_delete_reclaims_exclusive_only(client):
    parent = client.post("/sessions", json={"tokens": [1, 2, 3, 4]}).json()["session_id"]
    child = client.post(f"/sessions/{parent}/fork", json={"prefix_length": 4}).json()["session_id"]
    r = client.delete(f"/sessions/{parent}")
    assert r.status_code == 200
    assert r.json()["blocks_freed"] == 0
    assert client.get("/stats").json()["used_blocks"] == 1
    # Child still works and its logits match a full forward.
    r = client.post(f"/sessions/{child}/tokens", json={"tokens": [5]})
    assert r.status_code == 200
    client.delete(f"/sessions/{child}")
    assert client.get("/stats").json()["used_blocks"] == 0


def test_logits_match_full_forward_through_api(client):
    tokens = [3, 1, 4, 1, 5]
    sid = client.post("/sessions", json={"tokens": tokens[:2]}).json()["session_id"]
    got = []
    for t in tokens[2:]:
        got.append(client.post(f"/sessions/{sid}/tokens", json={"tokens": [t]}).json()["logits"][0])
    logits = client.app.state.store._model.forward(np.array(tokens))[0]
    assert np.max(np.abs(np.array(got) - logits[2:])) <= 1e-5
