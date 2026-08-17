"""RAG loop helpers extracted from MultiAgentWorkflow (Task 8, Step 5).

Methods are relocated verbatim; behavior is identical.
"""

import contextlib
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import interrupt

from app.ai.rag_tool_actions import (
    canonicalize_rag_tool_call,
    execute_search_documents_action,
    fit_rag_tool_message_content,
)
from app.ai.schemas import AgentMessage, AgentResponse, GraphState, MessageRole
from app.ai.token_counter import TokenCounter
from app.ai.tool_context import (
    rich_response_capable_from_context,
    tool_execution_context,
)
from app.ai.tool_execution import (
    apply_tool_output_offload,
    build_rejected_tool_artifacts,
    build_tool_artifact,
    ensure_agent_tool_map,
    execute_tool_calls,
)
from app.ai.utils import apply_hitl_decisions, make_json_safe, normalize_tool_call
from app.core.config import settings
from app.observability.rag import rag_metrics
from app.services.rag_evidence import EvidencePack
from app.services.rag_grounding import (
    GroundedAnswerGate,
    evidence_pack_from_payloads,
    parse_grounded_answer,
    render_grounded_answer,
)

logger = logging.getLogger("app.ai.graph")
_apply_decisions = apply_hitl_decisions

# Control-plane results whose model-visible text is server-generated, bounded by
# construction, and re-read after execution. ``_apply_hand_off_if_present``
# parses ``hand_off`` output and rewrites it in place with refusal feedback, so
# replacing it with a budget omission marker would both hide the delegation and
# discard the refusal the model needs in order to stop retrying. They are still
# charged against the cumulative allowance, just never replaced.
_UNBOUNDABLE_TOOL_NAMES = frozenset({"hand_off"})


class RagLoopMixin:
    """Relocated rag_loop methods for :class:`MultiAgentWorkflow`."""

    @staticmethod
    def _consume_evidence_token_counter(
        agent: Any,
        metadata: dict[str, Any],
        *,
        provider: str,
        model: str,
    ) -> TokenCounter:
        # Never allow a legacy live object to propagate farther through state.
        metadata.pop("_evidence_token_counter", None)
        descriptor = metadata.pop("evidence_tokenization", None)
        resolver = getattr(agent, "_take_evidence_token_counter", None)
        if callable(resolver):
            try:
                return resolver(descriptor, provider=provider, model=model)
            except Exception:
                logger.exception("Failed to resolve ephemeral evidence counter")
        return TokenCounter()

    @staticmethod
    def _discard_evidence_token_counter(agent: Any, metadata: dict[str, Any]) -> None:
        metadata.pop("_evidence_token_counter", None)
        descriptor = metadata.pop("evidence_tokenization", None)
        discard = getattr(agent, "_discard_evidence_token_counter", None)
        if callable(discard):
            discard(descriptor)

    def _grounded_answer_gate(self) -> GroundedAnswerGate:
        gate = getattr(self.rag_agent, "grounded_answer_gate", None)
        if isinstance(gate, GroundedAnswerGate):
            return gate
        return GroundedAnswerGate(
            getattr(settings, "min_citation_coverage", 0.5),
            metrics=rag_metrics,
        )

    def _current_turn_evidence(self, state: GraphState, messages: list[Any]) -> EvidencePack:
        """Merge only the evidence packs this turn's own tool calls produced.

        Ids are reissued per pack, so an id from an earlier turn is not a
        citation this turn can authorize.
        """
        last_human_idx = self._find_last_human_message_index(messages)
        start = 0 if last_human_idx is None else last_human_idx + 1
        turn_tool_call_ids = {
            str(tool_call.get("id") or "")
            for message in messages[start:]
            if isinstance(message, AIMessage) and message.tool_calls
            for tool_call in message.tool_calls
            if isinstance(tool_call, dict)
        }
        context = state.get("context") or {}
        payloads = [
            artifact["rag_evidence"]
            for artifact in (context.get("tool_artifacts") or [])
            if isinstance(artifact, dict)
            and isinstance(artifact.get("rag_evidence"), dict)
            and str(artifact.get("tool_call_id") or "") in turn_tool_call_ids
        ]
        return evidence_pack_from_payloads(payloads)

    def _grounded_regenerator(
        self,
        state: GraphState,
        question: str,
        evidence: EvidencePack,
    ) -> Any | None:
        regenerate = getattr(self.rag_agent, "regenerate_grounded_answer", None)
        if not callable(regenerate):
            return None

        async def _regenerate(*, reason_codes: Any):
            return await regenerate(
                question=question,
                evidence=evidence,
                reason_codes=reason_codes,
                conversation_id=state.get("conversation_id"),
                user_id=state.get("user_id"),
                model_request=state.get("model_request"),
            )

        return _regenerate

    async def _apply_grounded_answer_gate(
        self,
        state: GraphState,
        response: AgentResponse,
        *,
        question: str,
    ) -> AgentResponse:
        """Validate one final RAG answer against the current turn's evidence.

        Enforcement is opt-in. With ``rag_grounded_answer_gate_enabled`` off the
        model's answer is returned unchanged and only the shadow validation
        record is attached. The live evidence token counter is already consumed
        or discarded before this runs, and the regeneration is a tool-free pass
        that registers no descriptor, so this exit stays leak-free.
        """
        if not getattr(settings, "enable_citation_verification", False):
            return response
        if getattr(response, "error", None):
            # A failed turn reports its own error. It is not an answer to ground,
            # and replacing it would hide the failure from the user.
            return response
        text = str(response.message.content or "")
        if not text.strip():
            return response

        messages = state.get("messages", []) or []
        evidence = self._current_turn_evidence(state, messages)
        enforced = bool(getattr(settings, "rag_grounded_answer_gate_enabled", False))
        finalization = await self._grounded_answer_gate().finalize_answer(
            question=question,
            evidence=evidence,
            answer=parse_grounded_answer(text),
            regenerate=(
                self._grounded_regenerator(state, question, evidence) if enforced else None
            ),
            mode="enforced" if enforced else "shadow",
        )
        metadata = response.metadata if isinstance(response.metadata, dict) else {}
        metadata["grounded_answer"] = finalization.to_metadata()
        response.metadata = metadata
        if not enforced:
            return response

        # A regenerated answer has no original prose to preserve, and an
        # abstention replaces it outright; otherwise keep the model's formatting
        # and let the server append the citation block.
        keep_prose = not finalization.answer.abstained and not finalization.regenerated
        response.message.content = render_grounded_answer(
            finalization.answer,
            evidence,
            text=text if keep_prose else None,
        )
        return response

    async def _rag_node(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        conversation_history = await self._get_conversation_history(
            conversation_id, user_id, agent_key="rag", state=state
        )
        device_id = state.get("device_id")

        last_human_idx = self._find_last_human_message_index(messages)
        original_query = messages[last_human_idx].content if last_human_idx is not None else content

        rag_tool_messages = []
        if last_human_idx is not None:
            for msg in messages[last_human_idx + 1 :]:
                if isinstance(msg, ToolMessage):
                    # ``hand_off`` ToolMessages are control-plane signals, not
                    # RAG evidence — skip them so handoff JSON does not leak
                    # into the delegated agent's tool context.
                    if getattr(msg, "name", None) == "hand_off":
                        continue
                    rag_tool_messages.append(msg)
                elif isinstance(msg, AIMessage) and msg.tool_calls:
                    rag_tool_messages.append(msg)

        rag_context = state.get("context", {}) or {}
        metadata = {
            "persona": state.get("persona"),
            "history": conversation_history,
            "original_query": original_query,
            "rag_tool_messages": rag_tool_messages,
            "agentic_images": rag_context.get(
                "agentic_images", []
            ),  # Pass images for multimodal LLM
            "model_request": state.get("model_request"),
            "user_id": user_id,
            "device_id": device_id,
        }
        # When ``_should_continue_rag`` decides the budget is exhausted, it
        # sets these flags on the context so this RAG turn becomes a no-tools
        # synthesis pass. Forward them into the AgentMessage metadata —
        # ``RAGAgent._process_message_agentic`` reads them.
        if rag_context.get("rag_force_final_response"):
            metadata["rag_force_final_response"] = True
            notice = rag_context.get("rag_tool_budget_notice")
            if notice:
                metadata["rag_tool_budget_notice"] = notice

        agent_msg = AgentMessage(
            role=MessageRole.USER,
            content=original_query,
            metadata=metadata,
            attachments=self._get_state_attachments(state),
        )

        multi_agent_kwargs = self._multi_agent_kwargs(state, "rag_agent")
        response = await self.rag_agent.process_message(
            agent_msg,
            conversation_id,
            internal_tools=multi_agent_kwargs.get("internal_tools"),
            handoff_target_descriptions=multi_agent_kwargs.get("handoff_target_descriptions"),
        )
        response = self._finalize_forced_final_response(state, response)
        if not response.message.tool_calls:
            self._discard_evidence_token_counter(
                self.rag_agent,
                response.metadata or {},
            )
            response = await self._apply_grounded_answer_gate(
                state,
                response,
                question=str(original_query or ""),
            )
        self._merge_tool_artifacts(state, response)
        state["response"] = response

        ai_kwargs = {"content": response.message.content or ""}
        if response.message.tool_calls:
            ai_kwargs["tool_calls"] = response.message.tool_calls
        state.setdefault("messages", []).append(AIMessage(**ai_kwargs))

        return state

    async def _rag_tools_node(self, state: GraphState) -> GraphState:
        """
        Execute RAG document exploration tools (agentic mode).

        - SCAN_ALL: Preview all documents in conversation
        - READ_DOCUMENT: Full content of specific document
        - SEARCH_CHUNKS: Vector search
        - GREP_DOCUMENT: Regex search
        - LIST_DOCUMENTS: List available documents
        """
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            # ``_should_call_rag_tools`` should never route here without a
            # tool-calling assistant message, but this node is also the only
            # consumer of the one-shot counter reference. Release it rather than
            # leaving it parked in the store until its TTL.
            self._discard_evidence_token_counter(
                self.rag_agent,
                getattr(state.get("response"), "metadata", None) or {},
            )
            return state

        conversation_id = state.get("conversation_id")
        user_id = state.get("user_id")
        tool_outputs = []
        context = state.get("context", {})
        tool_artifacts: list[dict[str, Any]] = []
        all_images: list[dict[str, str]] = []
        max_agentic_images = getattr(settings, "agentic_rag_max_images", 6)
        response = state.get("response")
        response_metadata = getattr(response, "metadata", {}) or {}
        request_budget = response_metadata.get("request_budget") or {}
        raw_evidence_allowance = request_budget.get("evidence_token_allowance")
        # Evidence packs fail closed on a missing allowance (zero tokens), but
        # non-pack results must not: "absent" means the request budget never ran,
        # so there is no authoritative remainder to enforce against.
        evidence_allowance_authoritative = raw_evidence_allowance is not None
        remaining_evidence_allowance = max(
            0,
            int(raw_evidence_allowance if evidence_allowance_authoritative else 0),
        )
        evidence_provider = str(response_metadata.get("provider") or "gemini")
        evidence_model = str(response_metadata.get("model") or "gemini-2.5-flash")
        evidence_token_counter = self._consume_evidence_token_counter(
            self.rag_agent,
            response_metadata,
            provider=evidence_provider,
            model=evidence_model,
        )

        def enforceable_allowance(name: Any) -> int | None:
            """Remainder one result may be bounded to, or None when unbounded."""
            if not evidence_allowance_authoritative:
                return None
            if str(name or "") in _UNBOUNDABLE_TOOL_NAMES:
                return None
            return remaining_evidence_allowance

        last_human_idx = self._find_last_human_message_index(messages)
        question = (
            str(messages[last_human_idx].content)
            if last_human_idx is not None
            else ""
        )

        # Track agentic iteration count
        agentic_iteration = context.get("agentic_rag_iteration", 0) + 1
        context["agentic_rag_iteration"] = agentic_iteration

        normalized_tool_calls = [
            canonicalize_rag_tool_call(normalize_tool_call(tool_call))
            for tool_call in last_message.tool_calls
        ]

        non_search_tool_calls = [
            tc for tc in normalized_tool_calls if tc.get("name") != "search_documents"
        ]
        non_search_outputs_by_id: dict[str, dict[str, Any]] = {}
        rejected_feedback: dict[str, str] = {}
        selected_agent_name = state.get("selected_agent")
        agent = self.agents.get(selected_agent_name) if selected_agent_name else None
        handoff_tool = self._handoff_tool_for_agent(state, selected_agent_name)
        scoped_internal_tools = [handoff_tool] if handoff_tool else None

        if non_search_tool_calls:
            tool_calls_to_execute = list(non_search_tool_calls)

            if await self._needs_approval(
                state,
                non_search_tool_calls,
                agent=agent,
                internal_tools=scoped_internal_tools,
            ):
                interrupt_payload = await self._prepare_interrupt_payload(
                    state,
                    tool_calls=non_search_tool_calls,
                    agent=agent,
                    internal_tools=scoped_internal_tools,
                )
                human_decisions = interrupt(interrupt_payload)

                if not human_decisions:
                    tool_calls_to_execute, rejected_feedback = _apply_decisions(
                        non_search_tool_calls, []
                    )
                else:
                    tool_calls_to_execute, rejected_feedback = _apply_decisions(
                        non_search_tool_calls, human_decisions
                    )

                for tc_id, feedback in rejected_feedback.items():
                    non_search_outputs_by_id[tc_id] = {"content": feedback}

            if rejected_feedback:
                tool_artifacts.extend(
                    build_rejected_tool_artifacts(
                        tool_calls=non_search_tool_calls,
                        rejected_feedback=rejected_feedback,
                    )
                )

            if tool_calls_to_execute:
                tool_map = (
                    await ensure_agent_tool_map(
                        agent,
                        conversation_id=conversation_id,
                        user_id=user_id,
                        device_id=state.get("device_id"),
                        internal_tools=scoped_internal_tools,
                    )
                    if agent
                    else {}
                )

                # Extract context for tool execution. Use the deferred-state key
                # (tool_state_key) so custom agents resolve their own loaded tools.
                user_id = state.get("user_id")
                if agent:
                    agent_key = (
                        getattr(agent, "tool_state_key", None)
                        or getattr(agent, "agent_config_key", None)
                        or "rag"
                    )
                else:
                    agent_key = "rag"
                device_id = state.get("device_id")

                # Execute tools with context set for deferred tool loading support
                with tool_execution_context(
                    conversation_id,
                    user_id,
                    agent_key,
                    device_id,
                    rich_response_capable=rich_response_capable_from_context(
                        state.get("context")
                    ),
                ):
                    outputs, artifacts, images = await execute_tool_calls(
                        tool_calls=tool_calls_to_execute,
                        tool_map=tool_map,
                        capture_images=True,
                        device_id=device_id,
                        agent=agent,
                        conversation_id=conversation_id,
                        user_id=user_id,
                    )
                for output in outputs:
                    if output.get("tool_call_id"):
                        non_search_outputs_by_id[output["tool_call_id"]] = output
                tool_artifacts.extend(artifacts)
                all_images.extend(images)

        for tool_call_data in normalized_tool_calls:
            tool_name = tool_call_data.get("name")
            tool_id = tool_call_data.get("id")
            tool_args = tool_call_data.get("args", {})

            if tool_name != "search_documents":
                stored = non_search_outputs_by_id.get(tool_id)
                entry: dict[str, Any] = {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                }
                if stored is not None:
                    entry["content"] = stored.get("content", "")
                    render = stored.get("render")
                    if isinstance(render, dict):
                        entry["render"] = render
                else:
                    entry["content"] = f"Error: Tool {tool_name} not found"
                fitted_content, consumed_tokens, budget_omitted = (
                    fit_rag_tool_message_content(
                        content=entry["content"],
                        allowance=enforceable_allowance(tool_name),
                        token_counter=evidence_token_counter,
                        provider=evidence_provider,
                        model=evidence_model,
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                    )
                )
                entry["content"] = fitted_content
                if budget_omitted:
                    entry["model_output_omitted"] = True
                tool_outputs.append(entry)
                remaining_evidence_allowance = max(
                    0,
                    remaining_evidence_allowance - consumed_tokens,
                )
                continue

            result, _, evidence = await execute_search_documents_action(
                rag_agent=self.rag_agent,
                conversation_id=conversation_id,
                tool_args=tool_args,
                context=context,
                max_agentic_images=max_agentic_images,
                user_id=state.get("user_id"),
                question=question,
                evidence_max_tokens=remaining_evidence_allowance,
                evidence_provider=evidence_provider,
                evidence_model=evidence_model,
                evidence_token_counter=evidence_token_counter,
            )

            parsed_error: dict[str, Any] | None = None
            if isinstance(result, str):
                with contextlib.suppress(Exception):
                    candidate = json.loads(result)
                    if isinstance(candidate, dict) and candidate.get("status") == "error":
                        parsed_error = candidate
            error = result if parsed_error or result.startswith("Error") else None
            if evidence.get("records") is not None:
                public_text, blob_info = result, None
                consumed_tokens = int(evidence.get("token_count") or 0)
                budget_omitted = False
            else:
                public_text, blob_info = apply_tool_output_offload(
                    output_text=result,
                    tool_call_id=tool_id,
                    tool_name=tool_name,
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                public_text, consumed_tokens, budget_omitted = (
                    fit_rag_tool_message_content(
                        content=public_text,
                        allowance=enforceable_allowance(tool_name),
                        token_counter=evidence_token_counter,
                        provider=evidence_provider,
                        model=evidence_model,
                        tool_call_id=tool_id,
                        tool_name=tool_name,
                    )
                )
            remaining_evidence_allowance = max(
                0,
                remaining_evidence_allowance - consumed_tokens,
            )
            artifact = build_tool_artifact(
                tool_call_id=tool_id,
                tool_name=tool_name,
                tool_args=tool_args,
                output_text=public_text,
                error=error,
            )
            if budget_omitted:
                artifact.update(
                    {
                        "model_output_omitted": True,
                        "model_output_omitted_reason": "context_budget",
                        "original_output_chars": len(str(result or "")),
                    }
                )
            if parsed_error:
                artifact["error_type"] = parsed_error.get("error_type")
                artifact["retryable"] = bool(parsed_error.get("retryable"))
            if blob_info:
                artifact.update(blob_info)
            if evidence:
                artifact["rag_evidence"] = make_json_safe(evidence)
            tool_artifacts.append(artifact)
            tool_outputs.append(
                {
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": public_text,
                }
            )

        # Interpret delegation before persisting ToolMessages so rejection
        # feedback replaces the matching result rather than adding a duplicate.
        # This rewrites entries of ``tool_outputs`` in place, so it must receive
        # that exact list — not a copy — and the entries it can rewrite must not
        # have been budget-replaced above (see ``_UNBOUNDABLE_TOOL_NAMES``).
        state = self._apply_hand_off_if_present(state, tool_outputs)

        # Add tool messages to state
        for output in tool_outputs:
            state.setdefault("messages", []).append(
                ToolMessage(
                    content=output["content"],
                    tool_call_id=output["tool_call_id"],
                    name=output["name"],
                )
            )

        state["context"] = context

        # Increment iteration count
        current_iteration = state.get("iteration_count") or 0
        state["iteration_count"] = current_iteration + 1

        if tool_artifacts:
            self._lift_rich_candidates(context, tool_artifacts)
            existing_artifacts = context.get("tool_artifacts", [])
            existing_artifacts.extend(tool_artifacts)
            context["tool_artifacts"] = existing_artifacts
        render_results = dict(context.get("tool_render_results", {}))
        for output in tool_outputs:
            tool_call_id = output.get("tool_call_id")
            render = output.get("render")
            if tool_call_id and isinstance(render, dict):
                render_results[str(tool_call_id)] = make_json_safe(render)
        if render_results:
            context["tool_render_results"] = render_results
        if all_images:
            existing_images = context.get("tool_images", [])
            existing_images.extend(all_images)
            context["tool_images"] = existing_images

        state["context"] = context
        self._update_tool_error_streak(state, tool_artifacts)

        return state

    def _should_call_rag_tools(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            context = state.get("context", {}) or {}
            if context.get("rag_force_final_response"):
                response = state.get("response")
                response_metadata = getattr(response, "metadata", {}) or {}
                self._discard_evidence_token_counter(
                    self.rag_agent,
                    response_metadata,
                )
                logger.warning(
                    "RAG final no-tools pass still emitted tool calls; ending "
                    "instead of executing more RAG tools."
                )
                return "end"
            return "rag_tools"

        return "end"

    def _should_continue_rag(self, state: GraphState) -> str:
        """
        Determine if RAG agentic loop should continue or end.

        When the iteration budget is reached, route the model back to
        ``rag_agent`` for one final no-tools synthesis pass so the user
        always sees an answer rather than a truncated tool log. If we have
        already forced that final pass once and the model still asked for
        tools, end the graph to avoid an infinite loop.
        """
        context = state.get("context", {})
        handoff = context.get("handoff") if isinstance(context, dict) else None
        if (
            isinstance(handoff, dict)
            and handoff.get("active")
            and handoff.get("source_agent") == "rag_agent"
            and handoff.get("target_agent") == state.get("selected_agent")
            and state.get("selected_agent") != "rag_agent"
        ):
            return self._route_target_for(state, state["selected_agent"])
        agentic_iteration = context.get("agentic_rag_iteration", 0)

        streak = context.get("tool_error_streak")
        if isinstance(streak, dict) and streak.get("count", 0) >= streak.get("limit", 3):
            if context.get("rag_force_final_response"):
                return "end"
            context["rag_force_final_response"] = True
            context["rag_tool_budget_notice"] = (
                "Repeated tool errors occurred. Produce the best final answer from the "
                "document evidence already available, explain the blocker briefly, and "
                "do not call tools."
            )
            state["context"] = context
            return "rag_agent"

        max_iterations = settings.agentic_max_iterations
        if agentic_iteration >= max_iterations:
            if context.get("rag_force_final_response"):
                logger.warning(
                    "RAG agentic loop already had its final no-tools pass "
                    "(iteration=%d, max=%d); ending to avoid an infinite loop.",
                    agentic_iteration,
                    max_iterations,
                )
                return "end"

            logger.warning(
                "RAG agentic loop reached max iterations (%d); forcing one "
                "final no-tools synthesis pass.",
                max_iterations,
            )
            context["rag_force_final_response"] = True
            context["rag_tool_budget_notice"] = (
                "The RAG tool budget is exhausted. Produce the final answer "
                "from the retrieved document evidence already available. "
                "Do not call tools."
            )
            state["context"] = context
            return "rag_agent"

        return "rag_agent"
