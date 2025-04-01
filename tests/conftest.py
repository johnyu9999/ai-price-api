# tests/conftest.py
import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client
