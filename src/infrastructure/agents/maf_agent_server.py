"""
MAF HTTP server entry point for Zava Customer Service Agent.

Exposes the Customer Service Agent as an HTTP REST endpoint using the
azure.ai.agentserver.agentframework package.  This enables:
  - AI Toolkit Agent Inspector for interactive testing
  - agentdev CLI for local debugging with breakpoints
  - Full streaming via Server-Sent Events
  - Future: containerised deployment to Azure AI Foundry

Uses MAF v1.0 pattern: Agent + OpenAIChatClient (Chat Completions / Responses
API).  No asst_XXX IDs are needed -- instructions and tools are passed directly
to the Agent constructor.

Usage:
    # HTTP server (default - works with Agent Inspector):
    python src/infrastructure/agents/maf_agent_server.py --server

    # CLI mode (simpler, for quick terminal testing):
    python src/infrastructure/agents/maf_agent_server.py --cli

    # Wrapped with debugpy + agentdev for VS Code debugging:
    python -m debugpy --listen 127.0.0.1:5679 \
           -m agentdev run src/infrastructure/agents/maf_agent_server.py \
           --verbose --port 8088 -- --server
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure the workspace root is on sys.path so `src.*` imports resolve whether
# this script is run directly (python maf_agent_server.py) or via agentdev.
_ROOT = Path(__file__).resolve().parents[3]  # .../Zava-AI-Logistics
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# override=False so a stale AZURE_OPENAI_API_KEY Windows User env var does
# not get promoted over a value already in the process environment from the
# deployed App Service settings.  The .env file is only consulted for values
# that are NOT already set.
from dotenv import load_dotenv
load_dotenv(override=False)

# ---------------------------------------------------------------------------
# MAF SDK (v1.0 pattern: Agent + OpenAIChatClient)
# ---------------------------------------------------------------------------
from agent_framework import (
    Agent,
    AgentResponseUpdate,
    Content,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)
from agent_framework.openai import OpenAIChatClient
from agent_framework.foundry import FoundryChatClient, FoundryMemoryProvider
from azure.identity.aio import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)

# NOTE: azure.ai.agentserver is imported lazily in run_server() to avoid its
# module-level init fetching an AZURE_OPENAI_API_KEY from the project discovery
# endpoint and polluting the environment, which breaks credential-based auth.

# ---------------------------------------------------------------------------
# Local tools and prompt
# ---------------------------------------------------------------------------
from src.infrastructure.agents.core.prompt_loader import get_agent_prompt
from src.infrastructure.agents.maf.tools import (
    search_parcels_by_recipient,
    track_parcel,
    search_parcels_by_driver,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_FOUNDRY_ENDPOINT: str = (
    os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    or os.getenv("FOUNDRY_PROJECT_ENDPOINT", "")
)
_MODEL: str = (
    os.getenv("FOUNDRY_MODEL_DEPLOYMENT_NAME")
    or os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
    or "gpt-4o"
)


def _credential():
    if os.getenv("USE_MANAGED_IDENTITY", "false").lower() == "true":
        return ManagedIdentityCredential()
    return DefaultAzureCredential(
        exclude_managed_identity_credential=True,
        exclude_visual_studio_code_credential=True,
        additionally_allowed_tenants=["*"],
    )


def _make_openai_client(middleware=None):
    """Build a chat client backed by Azure AD token credentials.

    Prefers FoundryChatClient when AZURE_AI_PROJECT_ENDPOINT is set (native
    Foundry auth, no api_version wiring needed).  Falls back to OpenAIChatClient
    with a direct Azure OpenAI endpoint for environments that only have
    AZURE_OPENAI_ENDPOINT configured.
    """
    if _FOUNDRY_ENDPOINT:
        kwargs: dict = {
            "project_endpoint": _FOUNDRY_ENDPOINT,
            "model": _MODEL,
            "credential": _credential(),
        }
        if middleware:
            kwargs["middleware"] = middleware
        return FoundryChatClient(**kwargs)

    token_provider = get_bearer_token_provider(
        _credential(), "https://cognitiveservices.azure.com/.default"
    )
    kwargs = {
        "azure_endpoint": _OPENAI_ENDPOINT,
        "model": _MODEL,
        "credential": token_provider,
        "api_version": "2025-03-01-preview",
    }
    if middleware:
        kwargs["middleware"] = middleware
    return OpenAIChatClient(**kwargs)


def _make_cs_agent(middleware=None) -> Agent:
    """Create the Customer Service Agent with tools, system prompt, and Foundry memory."""
    instructions = ""
    try:
        instructions = get_agent_prompt("customer-service")
    except Exception:
        pass

    extra: dict = {}
    if _FOUNDRY_ENDPOINT:
        try:
            extra["context_providers"] = [
                FoundryMemoryProvider(memory_store_name="zava-cs-memory")
            ]
        except Exception:
            pass  # degrade gracefully if the memory store has not been provisioned yet

    return Agent(
        client=_make_openai_client(middleware=middleware),
        name="zava-customer-service",
        instructions=instructions or None,
        tools=[track_parcel, search_parcels_by_recipient, search_parcels_by_driver],
        **extra,
    )


# ---------------------------------------------------------------------------
# Executor: wraps the Agent with the @handler contract
# ---------------------------------------------------------------------------

class CustomerServiceExecutor:
    """
    Wraps the Customer Service Agent as a MAF workflow executor.

    The @handler method is called by WorkflowBuilder for each incoming
    message.  It forwards the messages to the agent and streams the
    response back via ctx.yield_output().
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self.id = "zava-customer-service"

    @handler
    async def on_message(
        self,
        messages: list[Message],
        ctx: WorkflowContext,
    ) -> None:
        response = await self._agent.run(messages)
        await ctx.yield_output(
            AgentResponseUpdate(
                contents=[Content("text", text=str(response))],
                role="assistant",
                author_name=self.id,
            )
        )


# ---------------------------------------------------------------------------
# Build & serve
# ---------------------------------------------------------------------------

async def run_server() -> None:
    """Start the HTTP server for the Agent Inspector."""
    # Import lazily so module-level init in agentserver does not run during CLI mode.
    from azure.ai.agentserver.agentframework import from_agent_framework  # noqa: PLC0415

    if not _OPENAI_ENDPOINT:
        sys.exit(
            "❌  AZURE_OPENAI_ENDPOINT must be set.\n"
            "    Add it to your .env file and try again."
        )

    cs_agent = _make_cs_agent()
    executor = CustomerServiceExecutor(cs_agent)

    instructions = ""
    try:
        instructions = get_agent_prompt("customer-service")
    except Exception:
        pass

    workflow_agent = (
        WorkflowBuilder(start_executor=executor)
        .build()
        .as_agent(
            name="Zava Customer Service",
            instructions=instructions or None,
        )
    )
    print("✅  Zava Customer Service Agent ready — starting HTTP server …")
    await from_agent_framework(workflow_agent).run_async()


async def run_cli() -> None:
    """Interactive terminal loop - useful for quick smoke-tests."""
    if not _OPENAI_ENDPOINT:
        sys.exit("❌  AZURE_OPENAI_ENDPOINT must be set.")

    agent = _make_cs_agent()
    print("Zava Customer Service Agent  (type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue
        result = await agent.run(user_input)
        print(f"\nAgent: {result}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zava MAF Customer Service Agent server"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--server", action="store_true", help="Run as HTTP server (default)")
    group.add_argument("--cli", action="store_true", help="Run as interactive CLI")
    args = parser.parse_args()

    if args.cli:
        asyncio.run(run_cli())
    else:
        asyncio.run(run_server())
