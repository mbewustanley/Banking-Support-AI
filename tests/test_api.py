import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app

@pytest.fixture(autouse=True)
def mock_banking_support_service():
    with patch("src.api.BankingSupportService"):
        yield


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_grounded_answer():
    expected_result = {
        "answer": "A duplicate charge must be investigated. [S1]",
        "sources": [
            {
                "citation_id": "S1",
                "source": "card_fees.md",
                "document_id": "card_fees",
                "distance": 0.25,
            }
        ],
        "abstained": False,
        "error": None,
    }

    with patch("src.api.BankingSupportService") as service_class:
        service_class.return_value.ask.return_value = expected_result

        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"question": "What happens with duplicate charges?"},
            )

    assert response.status_code == 200
    assert response.json() == expected_result
    service_class.return_value.ask.assert_called_once_with(
        "What happens with duplicate charges?"
    )


def test_chat_endpoint_rejects_blank_question():
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "   "},
        )

    assert response.status_code == 422


def test_chat_endpoint_validates_request_length():
    with TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"question": "x" * 2001},
        )

    assert response.status_code == 422


def test_chat_endpoint_handles_service_failure():
    with patch("src.api.BankingSupportService") as service_class:
        service_class.return_value.ask.side_effect = RuntimeError(
            "Internal implementation details"
        )

        with TestClient(app) as client:
            response = client.post(
                "/chat",
                json={"question": "What happens with duplicate charges?"},
            )

    assert response.status_code == 500
    assert (
        "Internal implementation details" not in response.text
    )
    assert response.json()["detail"] == (
        "An internal error occurred while processing the question."
    )
