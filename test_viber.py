#!/usr/bin/env python3
"""
Unit tests for the Viber CLI AI Assistant.

These tests cover conversation management, error handling, and integration with local and remote models.
"""
import pytest
import typer
from typer.testing import CliRunner
import os
from unittest.mock import patch, MagicMock
import requests
from viber import app

runner = CliRunner()

def ollama_is_running():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def test_list_conversations_empty(monkeypatch):
    """
    Test that the CLI reports no active conversations when none exist.
    """
    result = runner.invoke(app, ["list-conversations"])
    assert "No active conversations." in result.output
    assert result.exit_code == 0

def test_start_ollama_conversation():
    """
    Test starting a conversation using the Ollama local model (if server is running).
    """
    if not ollama_is_running():
        pytest.skip("Ollama server not running on localhost:11434")
    # Try to use the default model (llama2 or tinyllama)
    result = runner.invoke(app, ["start", "ollama_conv", "--local", "--use-ollama", "--ollama-model", "llama2"], input="exit\n")
    assert "Started conversation 'ollama_conv'" in result.output
    assert "Ending conversation 'ollama_conv'" in result.output
    assert result.exit_code == 0

@pytest.mark.skipif(not os.environ.get("RUN_REMOTE_TESTS"), reason="Remote API tests skipped unless RUN_REMOTE_TESTS=1")
def test_remote_conversation_no_api_key(monkeypatch):
    """
    Test that the CLI errors if no API key is provided for remote model.
    """
    result = runner.invoke(app, ["start", "testconv", "--api-key", ""])
    # The CLI will print '[ERROR] No API key provided for remote model.' and abort
    assert result.exit_code == 1
    assert "[ERROR] No API key provided for remote model." in result.output or "Aborted." in result.output

@pytest.mark.skipif(not os.environ.get("RUN_REMOTE_TESTS"), reason="Remote API tests skipped unless RUN_REMOTE_TESTS=1")
def test_start_remote_conversation(monkeypatch):
    """
    Test starting a remote conversation with a mocked ChatOpenAI model.
    """
    with patch("viber.ChatOpenAI") as mock_chat:
        mock_chat.return_value.invoke.return_value = "Hello!"
        result = runner.invoke(app, ["start", "remconv", "--api-key", "dummy"] , input="exit\n")
        assert "Started conversation 'remconv'" in result.output
        assert "Ending conversation 'remconv'" in result.output
        assert result.exit_code == 0

def test_local_conversation_no_model_path():
    """
    Test that the CLI errors if no model path is provided for local model.
    """
    result = runner.invoke(app, ["start", "localconv", "--local"])
    assert "You must provide a valid --model-path to a local Llama model file." in result.output
    assert result.exit_code != 0

def test_local_conversation_invalid_model_path():
    """
    Test that the CLI errors if an invalid model path is provided for local model.
    """
    result = runner.invoke(app, ["start", "localconv", "--local", "--model-path", "not_a_real_file.gguf"])
    assert "You must provide a valid --model-path to a local Llama model file." in result.output
    assert result.exit_code != 0

def test_start_local_conversation(monkeypatch, tmp_path):
    """
    Test starting a local conversation with a dummy Llama model file and mocked LlamaCpp.
    """
    # Create a dummy model file
    model_path = tmp_path / "dummy.gguf"
    model_path.write_text("")
    with patch("viber.LlamaCpp") as mock_llama:
        mock_llama.return_value.invoke.return_value = "Hi!"
        result = runner.invoke(app, ["start", "localconv", "--local", "--model-path", str(model_path)], input="exit\n")
        assert "Started conversation 'localconv'" in result.output
        assert "Ending conversation 'localconv'" in result.output
        assert result.exit_code == 0

@pytest.mark.skipif(not os.environ.get("RUN_REMOTE_TESTS"), reason="Remote API tests skipped unless RUN_REMOTE_TESTS=1")
def test_conversation_message_flow(monkeypatch):
    """
    Test the message flow in a remote conversation using mocked ChatOpenAI and RunnableWithMessageHistory.
    """
    with patch("viber.ChatOpenAI") as mock_chat, \
         patch("viber.RunnableWithMessageHistory") as mock_runnable:
        mock_runnable_instance = mock_runnable.return_value
        mock_runnable_instance.invoke.side_effect = ["AI response"]
        result = runner.invoke(app, ["start", "msgconv", "--api-key", "dummy"], input="hello\nexit\n")
        assert "AI response" in result.output
        assert result.exit_code == 0

@pytest.mark.skipif(not os.environ.get("RUN_REMOTE_TESTS"), reason="Remote API tests skipped unless RUN_REMOTE_TESTS=1")
def test_list_conversations_after_start(monkeypatch):
    """
    Test that after clearing the conversations dict, no active conversations are reported.
    """
    # Clear the global conversations dict for isolation
    from viber import conversations
    conversations.clear()
    result = runner.invoke(app, ["list-conversations"])
    assert "No active conversations." in result.output
    assert result.exit_code == 0
