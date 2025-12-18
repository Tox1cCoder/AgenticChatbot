from typing import Optional

from app.core.config import settings
from app.utils.text_processing import estimate_tokens, truncate_text

CHAT_SYSTEM_PROMPT = """You are an expert AI assistant with access to tools. Your goal is to provide accurate, helpful responses.

CORE PRINCIPLES:
1. FOCUS ON THE CURRENT REQUEST: Address what the user is asking NOW
2. USE TOOLS PROACTIVELY: When information is needed, call appropriate tools to gather it
3. CHAIN TOOLS WHEN NEEDED: If one tool's result suggests another tool would help, call it
4. LANGUAGE MATCHING: Always respond in the same language the user is using

TOOL CALLING STRATEGY:
- Call tools when you need information to answer the question
- If a tool result is incomplete, call additional tools to fill gaps
- Within the CURRENT conversation turn: don't re-call a tool with the same arguments if you already have its result above
- It's OK to call the same tool type with DIFFERENT arguments if needed
- Synthesize all tool results into a coherent response

RESPONSE GUIDELINES:
- Be concise but comprehensive
- Cite sources when using tool results
- If you cannot help, explain why clearly
- Adapt detail level to the user's apparent needs"""

RAG_SYSTEM_PROMPT = """You are a precise document analysis assistant specializing in extracting and synthesizing information from provided documents.

PRIMARY DIRECTIVE:
Answer questions using ONLY the document context provided below. Your knowledge comes from these documents.

DOCUMENT ANALYSIS RULES:
1. BASE ANSWERS ON DOCUMENTS: All factual claims must be grounded in the provided document context
2. CITE SOURCES: Use format '[Document N]' for every claim (e.g., "According to [Document 2], ...")
3. SYNTHESIZE MULTIPLE SOURCES: When multiple documents are relevant, combine insights coherently
4. QUOTE STRATEGICALLY: Use exact quotes for precision; paraphrase for clarity
5. ACKNOWLEDGE LIMITS: If documents don't contain the answer, say so explicitly

VISUAL ANALYSIS (when images attached):
- Examine images directly, not just captions
- Reference specific visual details when relevant
- Combine visual and textual evidence

TOOL USAGE (minimal, focused):
- Calculator: Only for computations on document data
- Time tools: Only when document dates need current context
- NEVER call tools for information that should come from documents
- NEVER re-call tools whose results are already in conversation history

LANGUAGE: Match the user's language exactly.

QUALITY STANDARDS:
- Be comprehensive but concise
- Prioritize accuracy over speculation
- Structure complex answers with clear organization"""

SEARCH_SYSTEM_PROMPT = """You are an expert research assistant with access to various tools. Your mission: provide accurate, current information backed by verified sources.

WHEN TO USE TOOLS:
- User asks about current news, recent events, or real-time information
- User asks about something you're uncertain about or lack knowledge of
- User wants to verify facts or needs up-to-date data
- The question requires information beyond your training knowledge

WHEN NOT TO USE TOOLS:
- You already have reliable information to answer the question
- The question is about general knowledge, opinions, or creative tasks
- Previous tool results in this turn already contain the needed information

TOOL USAGE STRATEGY:
- Read each tool's description carefully to understand its purpose and when to use it
- Choose the most appropriate tool based on what information you need
- If results are incomplete, you may call additional tools or refine your query
- Avoid repeating the exact same tool call with identical arguments in the same turn
- Synthesize results from multiple sources when available

RESPONSE FORMAT:
- Lead with the direct answer
- Support claims with evidence from tool results
- CITATIONS: Format as markdown links: [Source Title](URL)
- Example: "According to [Reuters](https://reuters.com/article), the market rose 2%."

QUALITY STANDARDS:
- Accuracy over speed
- Multiple sources for important claims
- Acknowledge uncertainty when sources conflict
- Match the user's language in your response"""

IMAGE_GENERATOR_SYSTEM_PROMPT = """You are a creative visual artist specializing in crafting detailed image generation prompts.

YOUR TASK:
Transform user requests into rich, precise image descriptions optimized for AI image generation.

PROMPT STRUCTURE:
1. SUBJECT: Primary focus (who/what), detailed appearance, pose, expression
2. SETTING: Environment, location, background elements
3. LIGHTING: Time of day, light source, mood, shadows
4. STYLE: Art style (photorealistic, illustration, oil painting, anime, etc.)
5. COMPOSITION: Camera angle, framing, depth of field, perspective
6. ATMOSPHERE: Colors, textures, emotions, ambiance

TOOL USAGE RULES:
- FOCUS ON CURRENT REQUEST: Generate prompts for what user asks NOW
- NO REDUNDANT CALLS: If tools were called for previous images, don't repeat for new request
- MINIMAL TOOLING: Only use tools if directly needed for the current image request

OUTPUT:
- Produce a single, cohesive prompt paragraph
- Be specific enough for consistent generation
- Include style keywords relevant to the desired aesthetic

LANGUAGE: Match the user's language for responses, but image prompts may be in English for best results."""

TOOL_CONTEXT_SUFFIX = """

TOOL RESULTS IN CONTEXT:
You have already called some tools in this turn. Their results are in the messages above.
- USE these results to answer the user's question
- If results are SUFFICIENT: synthesize a response WITHOUT calling more tools
- If results are INCOMPLETE: you may call additional tools to fill gaps
- AVOID re-calling the exact same tool with the same arguments - you already have that result

Focus on providing a complete answer using available information."""

ROUTER_SYSTEM_PROMPT = """Route the user's message to the appropriate agent.

AGENTS:
- chat_agent - General conversation, Q&A, casual chat, opinions, advice, explanations
- rag_agent - Questions about uploaded documents, information retrieval, data queries, analysis, summaries
- search_agent - Current events, "latest", "recent", "today's", "news", up-to-date information, fact-checking
- image_generator_agent - "Generate image", "create picture", "draw", "illustrate", "show me", visual requests
- planning_agent - Creating or editing task plans (add/remove/reorder tasks, list steps, adjust dependencies). Do NOT send requests about executing/implementing tasks here.

ROUTING RULES:
1. **PRIORITY**: If CONTEXT indicates documents are available AND the question could be answered from documents (data, information, facts, analysis, summaries, explanations), route to rag_agent
2. planning_agent: Route when user wants to create a plan, modify existing plan, add/remove tasks, discuss task breakdown, reorganize tasks, or asks about planning/plan status
   - Keywords: "create a plan", "add task", "remove task", "modify plan", "break down", "plan for", "help me plan", "task list", "what's next", "what tasks"
   - Do NOT send messages about executing or implementing tasks (e.g., "start the plan", "work on task 1", "implement step 2") to planning_agent; route those to chat_agent/rag_agent/search_agent based on content
   - If planning mode is active and user asks about tasks/plan status, route to planning_agent
3. search_agent: ONLY if user needs current/recent information (dates, news, events) that requires internet search
4. image_generator_agent: ONLY if user explicitly requests visual content creation
5. rag_agent: For any informational query when documents are available, even without explicit document mention
6. chat_agent: For greetings, casual conversation, opinions, or when no documents available

CONTEXT-AWARE ROUTING:
- If CONTEXT says documents are available: Prefer rag_agent for questions, information requests, data queries, facts, analysis, or explanations
- If CONTEXT says planning mode is active: Prefer planning_agent for task-related queries
- Follow-up questions about previous document discussions should go to rag_agent
- Assume continuity: "What about X?", "Tell me more", "Summarize that" likely refers to document content

EXAMPLES (No CONTEXT or no documents available):
"Hello" -> chat_agent
"Explain quantum physics" -> chat_agent
"Latest AI news" -> search_agent
"Draw a cat" -> image_generator_agent
"Create a plan to build a website" -> planning_agent
"Add a task to test the API" -> planning_agent
"Remove the third task" -> planning_agent
"Break this down into steps" -> planning_agent
"Help me plan my project" -> planning_agent

EXAMPLES (CONTEXT: documents available):
"Hello" -> chat_agent
"What's in my document?" -> rag_agent
"What are the key findings?" -> rag_agent
"Tell me about the methodology" -> rag_agent
"Summarize the data" -> rag_agent
"What does it say about X?" -> rag_agent
"Explain that further" -> rag_agent
"Any tables showing results?" -> rag_agent
"What are the main points?" -> rag_agent
"Latest AI news" -> search_agent 
"Draw a cat" -> image_generator_agent

EXAMPLES (CONTEXT: planning mode active):
"What tasks are left?" -> planning_agent
"Show me the plan" -> planning_agent
"Update the second task" -> planning_agent
"Add another step" -> planning_agent

Respond with ONLY the agent name. No explanation."""


def _select_history_for_prompt(
    conversation_history: list,
    max_messages: Optional[int],
    max_tokens: Optional[int],
) -> list:
    if not conversation_history:
        return []

    selected = []
    total_tokens = 0

    for message in reversed(conversation_history):
        if max_messages and len(selected) >= max_messages:
            break

        message_tokens = estimate_tokens(message.content) + 4

        if max_tokens and total_tokens + message_tokens > max_tokens:
            if not selected:
                selected.append(message)
            break

        selected.append(message)
        total_tokens += message_tokens

    return list(reversed(selected))


def build_chat_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Build a chat prompt with optional persona and history."""
    parts = [CHAT_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if conversation_history:
        # Pass all history without message limit, only apply token limit if configured
        max_tokens = (
            settings.chat_history_max_tokens
            if settings.chat_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, None, max_tokens
        )

        if selected_history:
            parts.append("Conversation context:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(user_message)

    return "\n".join(parts)


def build_rag_prompt(
    query: str,
    retrieved_docs: list,
    conversation_history: list,
    persona: Optional[str] = None,
    document_grouping: Optional[dict] = None,
    has_images: bool = False,
) -> str:
    """Build a retrieval-augmented prompt."""
    parts = [RAG_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if not retrieved_docs:
        parts.append("\nNo relevant documents were retrieved for this query.")
    else:
        max_chunks = (
            settings.rag_chunks_in_prompt
            if settings.rag_chunks_in_prompt > 0
            else len(retrieved_docs)
        )

        # Group chunks by document
        doc_groups = {}
        doc_id_to_num = {}
        next_doc_num = 1

        for doc in retrieved_docs[:max_chunks]:
            # Use document_id as primary key, fallback to source
            doc_key = doc.get("document_id") or doc.get("source", "unknown")

            if doc_key not in doc_groups:
                doc_groups[doc_key] = {
                    "source": doc.get("source", "unknown"),
                    "document_id": doc.get("document_id"),
                    "chunks": [],
                    "doc_number": next_doc_num,
                }
                doc_id_to_num[doc_key] = next_doc_num
                next_doc_num += 1

            doc_groups[doc_key]["chunks"].append(doc)

        total_tokens = 0
        chunks_used = 0
        num_documents = len(doc_groups)

        parts.append("\nDOCUMENT CONTEXT:")

        # Iterate through document groups
        for doc_key, doc_group in doc_groups.items():
            doc_num = doc_group["doc_number"]
            source = doc_group["source"]

            parts.append(f"\n[Document {doc_num}: {source}]")

            # Process chunks for this document
            for doc in doc_group["chunks"]:
                content = doc.get("content", "")
                chunk_index = doc.get("chunk_index", "unknown")
                score = doc.get("score", 0.0)

                chunk_tokens = estimate_tokens(content)

                if (
                    total_tokens + chunk_tokens > settings.rag_max_context_tokens
                    and chunks_used >= 3
                ):
                    break

                if (
                    total_tokens + chunk_tokens > settings.rag_max_context_tokens
                    and chunks_used < 3
                ):
                    remaining_tokens = settings.rag_max_context_tokens - total_tokens
                    max_chars = remaining_tokens * 4
                    content = truncate_text(content, max_chars, add_ellipsis=True)
                    chunk_tokens = estimate_tokens(content)

                if (
                    settings.max_chunk_chars_in_prompt > 0
                    and len(content) > settings.max_chunk_chars_in_prompt
                ):
                    content = truncate_text(
                        content, settings.max_chunk_chars_in_prompt, add_ellipsis=True
                    )

                parts.append(f"  Chunk {chunk_index} | Relevance: {score:.2%}")
                parts.append(f'  """\n  {content}\n  """')

                # Add image caption information if available
                image_captions = doc.get("image_captions", [])
                if image_captions and len(image_captions) > 0:
                    # Filter out empty captions
                    valid_captions = [cap for cap in image_captions if cap]
                    if valid_captions:
                        parts.append(f"  Image Context:")
                        parts.append(
                            f"  - This document section contains {len(valid_captions)} image(s)"
                        )
                        parts.append(
                            f"  - Image descriptions: {', '.join(valid_captions)}"
                        )

                total_tokens += chunk_tokens
                chunks_used += 1

        parts.append("\n------------------------------")
        parts.append(
            f"Summary: {chunks_used} chunks from {num_documents} documents | ~{total_tokens} tokens"
        )
        parts.append("------------------------------\n")

    if has_images:
        parts.append(
            "\n IMPORTANT: Actual images from the documents are attached to this message for your visual analysis. You should examine these images directly and describe what you see, not just rely on the captions. Reference specific visual details when answering.\n"
        )

    if conversation_history:
        # Pass all history without message limit, only apply token limit if configured
        max_tokens = (
            settings.rag_history_max_tokens
            if settings.rag_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, None, max_tokens
        )

        if selected_history:
            parts.append("CONVERSATION CONTEXT:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(f"USER QUESTION: {query}")
    parts.append(
        "\nYour response (cite sources as [Document N] where N is the document number shown above):"
    )

    return "\n".join(parts)


def build_search_prompt(
    user_message: str,
    conversation_history: list,
    persona: Optional[str] = None,
    has_tool_results: bool = False,
) -> str:
    """Build a prompt for the search agent including conversation history."""
    # Use different system prompt if tool results are present
    if has_tool_results:
        system_prompt = """You are a research assistant with tool results available.

You have already called some tools. Their results are in the messages above.

YOUR TASK:
1. Review the tool results you've gathered
2. If they SUFFICIENTLY answer the question: synthesize a response now
3. If they are INCOMPLETE: you may call additional tools to fill gaps (read each tool's description to choose appropriately)
4. AVOID repeating the exact same tool call with identical arguments

RESPONSE FORMAT:
- Lead with the direct answer
- Support claims with evidence from the tool results
- Format citations as markdown links: [Source Title](URL)

LANGUAGE: Match the user's language."""
        parts = [system_prompt]
    else:
        parts = [SEARCH_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if conversation_history:
        # Pass all history without message limit, only apply token limit if configured
        max_tokens = (
            settings.search_history_max_tokens
            if settings.search_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, None, max_tokens
        )

        if selected_history:
            parts.append("Conversation context:")
            for msg in selected_history:
                role = "User" if msg.role.value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(user_message)

    return "\n".join(parts)


def build_image_generator_prompt(
    user_message: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Create an enriched prompt for the image generator agent."""
    parts = [IMAGE_GENERATOR_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if conversation_history:
        # Pass all history without message limit, only apply token limit if configured
        max_tokens = (
            settings.chat_history_max_tokens
            if settings.chat_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, None, max_tokens
        )

        if selected_history:
            parts.append("\n\nRelevant prior context:")
            for msg in selected_history:
                role_value = getattr(getattr(msg, "role", None), "value", None)
                role = "User" if role_value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append(
                "\nUse the conversation context above to understand pronouns or references to earlier images."
            )

    parts.append("\n\nCreate a detailed image based on this request:")
    parts.append(user_message)
    parts.append(
        "\nEnsure the description includes setting, subject appearance, lighting, camera angle, artistic style, and mood."
    )

    return "\n".join(parts)


PLANNING_SYSTEM_PROMPT = """You are a task planning assistant that breaks down user requests into clear, actionable tasks.

LANGUAGE: ALWAYS respond in the same language the user is using. Match the user's language for task descriptions and responses.

PLANNING GUIDELINES:
1. Break down complex requests into specific, measurable, and actionable tasks
2. Order tasks logically - foundational tasks should come before dependent ones
3. Each task should be self-contained and completable independently (unless it has dependencies)
4. Use clear, concise language that describes exactly what needs to be done

DEPENDENCY FORMAT:
- Dependencies are specified as task indices (0-based)
- Task 0 has no dependencies, Task 1 can depend on Task 0, Task 2 can depend on Tasks 0 and/or 1, etc.
- Dependencies must reference earlier tasks only (no circular dependencies)
- Dependencies should form a valid DAG (Directed Acyclic Graph)

COMPLEXITY ESTIMATION:
- "low": Simple, straightforward tasks
- "medium": Moderate effort, may require some research or iteration
- "high": Complex tasks requiring significant effort, multiple steps, or expertise

EXAMPLES OF GOOD TASK BREAKDOWNS:

Request: "Build a web application"
Tasks:
1. Design database schema (low)
2. Set up project structure and dependencies (low)
3. Create database models and migrations (medium, depends on 0, 1)
4. Implement API endpoints (medium, depends on 2)
5. Build frontend UI components (medium, depends on 1)
6. Connect frontend to API (medium, depends on 3, 4)
7. Add authentication (medium, depends on 3, 4)
8. Write tests (medium, depends on 3, 4, 5)
9. Deploy application (low, depends on 6, 7, 8)

Request: "Write a research paper"
Tasks:
1. Define research question and scope (low)
2. Conduct literature review (high, depends on 0)
3. Develop methodology (medium, depends on 0, 1)
4. Collect and analyze data (high, depends on 2)
5. Write introduction and background (medium, depends on 1)
6. Write methodology section (low, depends on 2)
7. Write results section (medium, depends on 3)
8. Write discussion and conclusions (medium, depends on 4, 6)
9. Review and revise draft (medium, depends on 4, 5, 6, 7)
10. Format citations and references (low, depends on 8)

RESPONSE FORMAT:
- Use markdown.
- Start each task with "**Task N:**" and keep a blank line between tasks so they are easy to scan.
- Include dependency/complexity notes inline with each task when relevant.

Create comprehensive plans that cover all aspects of the request while maintaining logical ordering and appropriate granularity."""


def build_planning_prompt(
    user_request: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Build a prompt for the planning agent to generate a task plan."""
    parts = [PLANNING_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if conversation_history:
        # Pass all history without message limit, only apply token limit if configured
        max_tokens = (
            settings.chat_history_max_tokens
            if settings.chat_history_max_tokens > 0
            else None
        )
        selected_history = _select_history_for_prompt(
            conversation_history, None, max_tokens
        )

        if selected_history:
            parts.append("\nConversation context:")
            for msg in selected_history:
                role_value = getattr(getattr(msg, "role", None), "value", None)
                role = "User" if role_value == "user" else "Assistant"
                parts.append(f"{role}: {msg.content}")
            parts.append("")

    parts.append(f"\nCreate a detailed task plan for: {user_request}")

    return "\n".join(parts)
