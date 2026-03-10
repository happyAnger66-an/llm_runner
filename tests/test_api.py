"""
API 测试
"""

import pytest
from fastapi.testclient import TestClient

from llm_run.api import create_app


@pytest.fixture
def client():
    """不加载 engine 的测试客户端"""
    app = create_app(engine_path=None)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_empty(client):
    """未加载 engine 时返回空列表"""
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"
    assert r.json()["data"] == []
