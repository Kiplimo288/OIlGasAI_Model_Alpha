"""
test_inference.py
-----------------
Unit tests for the inference layer.
Uses mocked backends to avoid requiring actual model weights.
"""

import pytest
from unittest.mock import MagicMock, patch
from inference.chat import chat, chat_with_history, _local_inference, _api_inference


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_local_backend():
    """Mock local model backend."""
    import torch

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "formatted prompt"
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    # Mock tokenizer call
    inputs = MagicMock()
    inputs.__getitem__ = lambda self, key: MagicMock(shape=[1, 10])
    tokenizer.return_value = inputs

    # Mock model
    model = MagicMock()
    model.device = "cpu"
    fake_output = MagicMock()
    fake_output.__getitem__ = lambda self, idx: [0] * 20
    model.generate.return_value = [fake_output]

    tokenizer.decode.return_value = "Centrifugal compressors under Subpart W must report using emission factors."

    return {"mode": "local", "model": model, "tokenizer": tokenizer}


@pytest.fixture
def mock_api_backend():
    """Mock HF API backend."""
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = "EPA Subpart W requires annual reporting for compressors."
    client.chat_completion.return_value = response
    return {"mode": "api", "client": client}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChatAPI:
    def test_chat_api_basic(self, mock_api_backend):
        """Basic single-turn chat returns a non-empty string."""
        result = chat("What is Subpart W?", mock_api_backend)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_with_history_api(self, mock_api_backend):
        """Multi-turn chat includes history in messages."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ]
        result = chat_with_history(history + [{"role": "user", "content": "What is LDAR?"}], mock_api_backend)
        assert isinstance(result, str)

        # Verify history was passed to client
        call_args = mock_api_backend["client"].chat_completion.call_args
        messages = call_args.kwargs["messages"]
        assert any(m["content"] == "Hello" for m in messages)

    def test_unknown_backend_raises(self):
        """Unknown backend mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend mode"):
            chat("test", {"mode": "unknown"})

    def test_custom_system_prompt(self, mock_api_backend):
        """Custom system prompt is passed through correctly."""
        custom_prompt = "You are a pipeline integrity expert."
        chat("Test question", mock_api_backend, system_prompt=custom_prompt)

        call_args = mock_api_backend["client"].chat_completion.call_args
        messages = call_args.kwargs["messages"]
        system_msg = next(m for m in messages if m["role"] == "system")
        assert system_msg["content"] == custom_prompt


class TestDomainResponses:
    """Test that OilGasAI Model Alpha-specific topics route correctly."""

    DOMAIN_QUERIES = [
        "EPA Subpart W compressor reporting",
        "LDAR leak threshold valves NSPS OOOOa",
        "MIRA sensor drift correction 30 days",
        "OGMP 2.0 Level 5 measurement",
    ]

    @pytest.mark.parametrize("query", DOMAIN_QUERIES)
    def test_domain_queries_complete(self, query, mock_api_backend):
        """All domain queries complete without error."""
        result = chat(query, mock_api_backend)
        assert result is not None
        assert isinstance(result, str)
