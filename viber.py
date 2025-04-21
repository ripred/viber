#!/usr/bin/env python3
"""
Viber CLI AI Assistant

This module provides a command-line interface for starting and managing AI conversations using local or remote models via LangChain.
"""

import typer
from typing import Optional
import os
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = typer.Typer()

# Store conversations in memory (dict: name -> ConversationChain)
conversations = {}


def get_model(local: bool, api_key: Optional[str] = None, model_path: Optional[str] = None, use_ollama: bool = False, ollama_model: Optional[str] = None):
    if local:
        if use_ollama:
            model_name = ollama_model or os.getenv("OLLAMA_MODEL", "llama2")
            return OllamaLLM(model=model_name)
        if not model_path or not os.path.isfile(model_path):
            typer.echo("[ERROR] You must provide a valid --model-path to a local Llama model file.")
            raise typer.Exit(1)
        return LlamaCpp(model_path=model_path, n_ctx=2048)
    else:
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            typer.echo("[ERROR] No API key provided for remote model.")
            raise typer.Exit(1)
        return ChatOpenAI(openai_api_key=api_key)


@app.command()
def start(
    name: str = typer.Argument(..., help="Conversation name"),
    local: bool = typer.Option(False, help="Use local model (default: remote)"),
    api_key: Optional[str] = typer.Option(None, help="API key for remote model"),
    model_path: Optional[str] = typer.Option(None, help="Path to local Llama model file (required if --local and not using Ollama)"),
    use_ollama: bool = typer.Option(False, help="Use Ollama server for local model (default: False)"),
    ollama_model: Optional[str] = typer.Option(None, help="Ollama model name (default: llama2)")
):
    """
    Start a new conversation with a language model.

    Args:
        name (str): Conversation name.
        local (bool): Use local model if True, otherwise remote.
        api_key (Optional[str]): API key for remote model.
        model_path (Optional[str]): Path to local Llama model file.
        use_ollama (bool): Use Ollama server for local model.
        ollama_model (Optional[str]): Ollama model name.
    """
    model = get_model(local, api_key, model_path, use_ollama, ollama_model)
    # Use RunnableWithMessageHistory for conversation management (LangChain >=0.2.7)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{input}")
    ])
    class SimpleMemory:
        """
        Simple conversation memory backend for storing message history.
        """
        def __init__(self):
            """Initialize message history."""
            self._history = []
        def add_message(self, message):
            """Add a message to history.

            Args:
                message: Message object (HumanMessage or AIMessage).
            """
            self._history.append(message)
        def get_messages(self):
            """Get all messages in history.

            Returns:
                list: List of messages.
            """
            return self._history
        def clear(self):
            """Clear message history."""
            self._history = []
        @property
        def messages(self):
            """Get message history (property for compatibility)."""
            return self._history
    memory = SimpleMemory()
    convo = RunnableWithMessageHistory(model, lambda session_id: memory, input_messages_key="input")
    conversations[name] = convo
    typer.echo(f"[INFO] Started conversation '{name}'. Type messages below (type 'exit' to stop):")
    session_id = name
    while True:
        user_input = input(f"[{name}] You: ")
        if user_input.lower() in ("exit", "quit"):
            typer.echo(f"[INFO] Ending conversation '{name}'.")
            break
        # Add user message to memory
        memory.add_message(HumanMessage(content=user_input))
        # Run the model with message history
        response = convo.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        # Add AI response to memory
        memory.add_message(AIMessage(content=response))
        typer.echo(f"[{name}] AI: {response}")


@app.command()
def list_conversations():
    """
    List all active conversations.
    """
    if not conversations:
        typer.echo("No active conversations.")
        return
    typer.echo("Active conversations:")
    for name in conversations:
        typer.echo(f"- {name}")


if __name__ == "__main__":
    """Entry point for the Viber CLI application."""
    app()
