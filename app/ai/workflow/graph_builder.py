"""Graph topology builder for :class:`MultiAgentWorkflow`.

Extracted verbatim from ``MultiAgentWorkflow._build_graph`` (behavior-preserving).
The single ``self`` receiver is threaded through as ``workflow`` and the
checkpointer is passed explicitly so the workflow instance stays the sole
owner of node handlers and routing callbacks.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.ai.schemas import GraphState


def build_workflow_graph(workflow: Any, *, checkpointer: Any | None) -> Any:
    graph = StateGraph(GraphState)

    # Durable compaction work advances atomically with assistant persistence;
    # provider-backed processing stays off the request hot path.
    graph.add_node("route", workflow._route_node)
    graph.add_node("chat_agent", workflow._chat_node)
    graph.add_node("rag_agent", workflow._rag_node)
    graph.add_node("search_agent", workflow._search_node)
    graph.add_node("image_generator_agent", workflow._image_generator_node)
    graph.add_node("planning_agent", workflow._planning_node)
    graph.add_node("canvas_agent", workflow._canvas_node)
    # Single static node that multiplexes every runtime custom-agent id
    # (custom_agent:<uuid>). The graph is never rebuilt per conversation.
    graph.add_node("custom_agent", workflow._custom_agent_node)
    graph.add_node("planning_tools", workflow._planning_tools_node)
    graph.add_node("rag_tools", workflow._rag_tools_node)
    graph.add_node("approval", workflow._approval_node)
    graph.add_node("tools", workflow._tool_node)

    # START -> route directly. Long-term summarization no longer runs on
    # the streaming hot path — it is refreshed after the assistant turn
    # is persisted (see MessageService).
    graph.add_edge(START, "route")

    graph.add_conditional_edges(
        "route",
        workflow._should_continue,
        {
            "chat_agent": "chat_agent",
            "rag_agent": "rag_agent",
            "search_agent": "search_agent",
            "image_generator_agent": "image_generator_agent",
            "planning_agent": "planning_agent",
            "canvas_agent": "canvas_agent",
            "custom_agent": "custom_agent",
            "end": END,
        },
    )

    # Consolidate conditional edges for agents that use standard tool calling
    tool_calling_agents = [
        "chat_agent",
        "search_agent",
        "image_generator_agent",
        "canvas_agent",
        "custom_agent",
    ]
    for agent_name in tool_calling_agents:
        graph.add_conditional_edges(
            agent_name,
            workflow._should_call_tools,
            {
                "approval": "approval",
                "tools": "tools",
                "end": END,
            },
        )

    # RAG agent: direct to END if not agentic, or rag_tools loop if agentic
    graph.add_conditional_edges(
        "rag_agent",
        workflow._should_call_rag_tools,
        {
            "rag_tools": "rag_tools",
            "end": END,
        },
    )

    rag_tools_routing = {agent_name: agent_name for agent_name in workflow.agents}
    rag_tools_routing["custom_agent"] = "custom_agent"
    rag_tools_routing["end"] = END
    graph.add_conditional_edges(
        "rag_tools",
        workflow._should_continue_rag,
        rag_tools_routing,
    )

    # Planning agent ReAct loop: planning_agent → planning_tools → planning_agent OR end
    graph.add_conditional_edges(
        "planning_agent",
        workflow._should_call_planning_tools,
        {
            "planning_tools": "planning_tools",
            "end": END,
        },
    )

    # planning_tools fans out either back into the planning loop, ends the
    # turn, or — when the Planning Agent invoked hand_off — routes to the
    # delegated top-level agent. Wiring every agent here is necessary so
    # LangGraph accepts ``selected_agent`` as a valid return from
    # ``_should_continue_planning``.
    planning_tools_routing = {agent_name: agent_name for agent_name in workflow.agents}
    planning_tools_routing["custom_agent"] = "custom_agent"
    planning_tools_routing["end"] = END
    graph.add_conditional_edges(
        "planning_tools",
        workflow._should_continue_planning,
        planning_tools_routing,
    )

    graph.add_edge("approval", "tools")

    # Dynamic tool routing map based on agent registry
    # Include ALL agents (including rag_agent) so hand_off delegation works.
    tool_routing_map = {agent_name: agent_name for agent_name in workflow.agents}
    tool_routing_map["custom_agent"] = "custom_agent"
    tool_routing_map["end"] = END

    graph.add_conditional_edges(
        "tools",
        workflow._route_tool_output,
        tool_routing_map,
    )

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
