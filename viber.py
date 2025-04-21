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
    """
    Select and initialize the appropriate language model (local or remote).

    Args:
        local (bool): Whether to use a local model.
        api_key (Optional[str]): API key for remote models (OpenAI).
        model_path (Optional[str]): Path to local Llama model file.
        use_ollama (bool): Whether to use Ollama server for local model.
        ollama_model (Optional[str]): Name of the Ollama model to use.

    Returns:
        An instance of the selected language model (ChatOpenAI, LlamaCpp, or OllamaLLM).

    Raises:
        typer.Exit: If required parameters are missing or invalid.
    """
    if local:
        if use_ollama:
            # Use Ollama server for local model inference
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
    # Create a prompt template for the conversation
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{input}")
    ])

    class SimpleMemory:
        """
        Simple conversation memory backend for storing message history.
        Provides add, get, and clear operations for message history.
        """
        def __init__(self):
            """Initialize message history as an empty list."""
            self._history = []
        def add_message(self, message):
            """
            Add a message to the history.

            Args:
                message: Message object (HumanMessage or AIMessage).
            """
            self._history.append(message)
        def get_messages(self):
            """
            Get all messages currently in the history.

            Returns:
                list: List of message objects.
            """
            return self._history
        def clear(self):
            """
            Clear all messages from the history.
            """
            self._history = []
        @property
        def messages(self):
            """
            Property to access message history for compatibility.

            Returns:
                list: List of message objects.
            """
            return self._history

    # Set up conversation memory and chain
    memory = SimpleMemory()
    convo = RunnableWithMessageHistory(model, lambda session_id: memory, input_messages_key="input")
    conversations[name] = convo
    typer.echo(f"[INFO] Started conversation '{name}'. Type messages below (type 'exit' to stop):")
    session_id = name
    while True:
        # Prompt user for input
        user_input = input(f"[{name}] You: ")
        if user_input.lower() in ("exit", "quit"):
            # Exit conversation loop
            typer.echo(f"[INFO] Ending conversation '{name}'.")
            break
        # Add user message to memory
        memory.add_message(HumanMessage(content=user_input))
        # Run the model with message history and get AI response
        response = convo.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
        # Add AI response to memory
        memory.add_message(AIMessage(content=response))
        # Display AI response
        typer.echo(f"[{name}] AI: {response}")


@app.command()
def list_conversations():
    """
    List all active conversations currently stored in memory.

    Prints the names of all active conversations, or a message if none exist.
    """
    if not conversations:
        typer.echo("No active conversations.")
        return
    typer.echo("Active conversations:")
    for name in conversations:
        typer.echo(f"- {name}")


if __name__ == "__main__":
    # Entry point for the Viber CLI application.
    app()
