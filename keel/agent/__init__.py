"""Tool-calling agent: typed tools, policy at the tool boundary, approval queue for write tools."""

from keel.agent.approvals import ApprovalQueue
from keel.agent.loop import AgentLoop, AgentResult
from keel.agent.policy import Decision, Policy
from keel.agent.tools import (
    TicketBook,
    ToolContext,
    ToolError,
    ToolRegistry,
    calculator_tool,
    create_ticket_tool,
    default_registry,
    http_get_tool,
    search_docs_tool,
    sql_readonly_tool,
)

__all__ = [
    "AgentLoop",
    "AgentResult",
    "ApprovalQueue",
    "Decision",
    "Policy",
    "TicketBook",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "calculator_tool",
    "create_ticket_tool",
    "default_registry",
    "http_get_tool",
    "search_docs_tool",
    "sql_readonly_tool",
]
