#!/usr/bin/env python3
"""ANSA/META CodeRAG Agent — Terminal Chat Interface.

Usage:
    python app_cli.py
"""

import sys
from pathlib import Path

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import settings
from bin.agent import CodeRAGAgent, AgentConfig


def main():
    print()
    print("=" * 60)
    print("  SafetyAgent — ANSA/META CodeRAG Agent (Terminal)")
    print("=" * 60)
    print(f"  LLM Model  : {settings.llm.primary_model}")
    print(f"  API Base   : {settings.llm.api_base_url}")
    print()
    print("  Commands:")
    print("    /reset  — Reset conversation")
    print("    /stats  — Show usage stats")
    print("    /quit   — Exit")
    print("=" * 60)
    print()

    config = AgentConfig(
        max_tool_calls=settings.agent.max_tool_calls,
        max_context_messages=settings.agent.max_context_messages,
    )
    agent = CodeRAGAgent(config=config)
    print("  Agent ready. Ask anything about ANSA/META API.\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() == "/quit":
            print("\nGoodbye!")
            break
        elif user_input.lower() == "/reset":
            agent.reset()
            print("\n  [Conversation reset]\n")
            continue
        elif user_input.lower() == "/stats":
            usage = agent.get_usage_summary()
            print("\n  Usage Statistics:")
            for k, v in usage.items():
                print(f"    {k}: {v}")
            print()
            continue

        # Chat
        print()
        try:
            response = agent.chat(user_input)
            print(f"Agent > {response}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
