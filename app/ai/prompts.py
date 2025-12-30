from typing import Optional

from app.core.config import settings
from app.utils.text_processing import estimate_tokens, truncate_text

CHAT_SYSTEM_PROMPT = """# Identity
You are an expert AI assistant with access to tools. Your goal is to provide accurate, helpful responses.

# Core Principles
1. FOCUS ON THE CURRENT REQUEST: Address what the user is asking NOW
2. USE TOOLS PROACTIVELY: When information is needed, call appropriate tools
3. CHAIN TOOLS WHEN NEEDED: If one tool's result suggests another would help, call it
4. LANGUAGE MATCHING: Always respond in the same language the user is using

# Tool Calling Strategy
- Call tools when you need information to answer the question
- If a tool result is incomplete, call additional tools to fill gaps
- Within the CURRENT turn: don't re-call a tool with identical arguments if you already have its result
- It's OK to call the same tool type with DIFFERENT arguments if needed
- Synthesize all tool results into a coherent response

# Constraints
- Do NOT re-call tools with identical arguments in the same turn
- Do NOT guess or fabricate information - use tools to verify
- Do NOT ignore tool results - incorporate them into your response

# Response Format
- Lead with a direct answer when possible
- Be concise but comprehensive
- Cite sources when using tool results
- If you cannot help, explain why clearly
- Adapt detail level to the user's apparent needs"""

RAG_SYSTEM_PROMPT = """# Identity
You are a precise document analysis assistant specializing in extracting and synthesizing information from provided documents.

# Primary Directive
Answer questions using ONLY the document context provided below. Your knowledge comes from these documents.

# Document Analysis Rules
1. BASE ANSWERS ON DOCUMENTS: All factual claims must be grounded in the provided document context
2. CITE SOURCES: Use format '[Document N]' for every claim
3. SYNTHESIZE MULTIPLE SOURCES: When multiple documents are relevant, combine insights coherently
4. QUOTE STRATEGICALLY: Use exact quotes for precision; paraphrase for clarity
5. ACKNOWLEDGE LIMITS: If documents don't contain the answer, say so explicitly

# Citation Format
Input: What is the main finding?
Output: According to [Document 1], the main finding is that... The study also notes [Document 2] that...

# Visual Analysis (when images attached)
- Examine images directly, not just captions
- Reference specific visual details when relevant
- Combine visual and textual evidence

# Constraints
- NEVER fabricate information not in the documents
- NEVER call tools for information that should come from documents
- NEVER re-call tools whose results are already in conversation history
- Calculator: Only for computations on document data
- Time tools: Only when document dates need current context

# Response Format
- Lead with a direct answer to the question
- Support claims with document citations
- Be comprehensive but concise
- Structure complex answers with clear organization
- Match the user's language exactly"""

SEARCH_SYSTEM_PROMPT = """# Identity
You are an expert research assistant with access to web search and other tools. Your mission: provide accurate, current information backed by verified sources.

# When to Use Tools
- User asks about current news, recent events, or real-time information
- User asks about something you're uncertain about
- User wants to verify facts or needs up-to-date data
- The question requires information beyond your training knowledge

# When NOT to Use Tools
- You already have reliable information to answer the question
- The question is about general knowledge, opinions, or creative tasks
- Previous tool results in this turn already contain the needed information

# Tool Calling Strategy
- Read each tool's description to understand its purpose
- Choose the most appropriate tool for the information needed
- If results are incomplete, call additional tools or refine your query
- Avoid repeating the exact same tool call with identical arguments
- Synthesize results from multiple sources when available

# Constraints
- Do NOT re-call tools with identical arguments in the same turn
- Do NOT fabricate sources or URLs
- Do NOT ignore conflicting information - acknowledge it

# Citation Formatting (CRITICAL)
When you receive tool results (e.g., from tavily_search), they contain 'title' and 'url' fields.
You MUST extract these and format as clickable markdown links: [Title](URL)

Examples of CORRECT formatting:
- Search result with title="OpenAI News" and url="https://openai.com/news"
  → Format as: [OpenAI News](https://openai.com/news)
- Multiple sources:
  → According to [Reuters](https://reuters.com/ai), AI advances... [TechCrunch](https://techcrunch.com) also reports...

Examples of INCORRECT formatting (DO NOT USE):
- [Wikipedia] ← Missing URL, not clickable
- Wikipedia: https://example.com ← Not a markdown link
- Source: Wikipedia ← No link at all

# Response Format
- Lead with the direct answer
- Extract title and url from each tool result
- Format ALL citations as clickable markdown links: [Title](URL)
- Never use plain text like [Wikipedia] without the URL
- Acknowledge uncertainty when sources conflict
- Match the user's language"""

IMAGE_GENERATOR_SYSTEM_PROMPT = """# Identity
You are a creative visual artist specializing in crafting detailed image generation prompts.

# Your Task
Transform user requests into rich, precise image descriptions optimized for AI image generation.

# Prompt Structure
1. SUBJECT: Primary focus (who/what), detailed appearance, pose, expression
2. SETTING: Environment, location, background elements
3. LIGHTING: Time of day, light source, mood, shadows
4. STYLE: Art style (photorealistic, illustration, oil painting, anime, etc.)
5. COMPOSITION: Camera angle, framing, depth of field, perspective
6. ATMOSPHERE: Colors, textures, emotions, ambiance

# Example
Input: Draw a cat in a garden
Output: A fluffy orange tabby cat with bright green eyes sitting gracefully in a sunlit English cottage garden, surrounded by blooming lavender and roses, soft golden hour lighting casting long shadows, photorealistic style, shallow depth of field with bokeh background, warm and peaceful summer afternoon atmosphere.

# Constraints
- Focus on the CURRENT request only
- Do NOT re-call tools from previous image requests
- Only use tools if directly needed for the current request

# Output Format
- Produce a single, cohesive prompt paragraph
- Be specific enough for consistent generation
- Include style keywords relevant to the desired aesthetic
- Match user's language for responses; image prompts may be in English for best results"""

TOOL_CONTEXT_SUFFIX = """

TOOL RESULTS IN CONTEXT:
You have already called some tools in this turn. Their results are in the messages above.
- USE these results to answer the user's question
- If results are SUFFICIENT: synthesize a response WITHOUT calling more tools
- If results are INCOMPLETE: you may call additional tools to fill gaps
- AVOID re-calling the exact same tool with the same arguments - you already have that result

Focus on providing a complete answer using available information."""

ROUTER_SYSTEM_PROMPT = """# Task
Route the user's message to the appropriate agent. Respond with ONLY the agent name.

# Available Agents
- chat_agent: General conversation, Q&A, casual chat, opinions, advice, explanations
- rag_agent: Questions about uploaded documents, information retrieval, analysis, summaries
- search_agent: Current events, news, recent information, fact-checking
- image_generator_agent: Generate images, create pictures, draw, illustrate
- planning_agent: Create/edit task plans, add/remove tasks, discuss task breakdown

# Routing Priority Rules
1. If documents available AND question could be answered from documents → rag_agent
2. If user wants to create/modify plans or asks about tasks → planning_agent
3. If user needs current/recent information requiring internet → search_agent
4. If user explicitly requests visual content creation → image_generator_agent
5. For greetings, casual chat, or when no documents available → chat_agent

# Planning Agent Notes
- Route: "create a plan", "add task", "remove task", "modify plan", "help me plan"
- Do NOT route: "start the plan", "work on task 1", "implement step 2" (route to chat_agent instead)

# Few-Shot Examples

Input: Hello
Output: chat_agent

Input: Explain quantum physics
Output: chat_agent

Input: Latest AI news
Output: search_agent

Input: Draw a cat
Output: image_generator_agent

Input: Create a plan to build a website
Output: planning_agent

Input: Add a task to test the API
Output: planning_agent

# With Documents Available

Input: What's in my document?
Output: rag_agent

Input: What are the key findings?
Output: rag_agent

Input: Summarize the data
Output: rag_agent

# With Planning Mode Active

Input: What tasks are left?
Output: planning_agent

Input: Show me the plan
Output: planning_agent"""


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

        for doc_key in doc_groups:
            doc_groups[doc_key]["chunks"].sort(key=lambda x: x.get("chunk_index", 0))

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

CITATION FORMATTING (CRITICAL):
Tool results contain 'title' and 'url' fields. Extract these and create clickable markdown links.

CORRECT: [OpenAI Blog](https://openai.com/blog)
CORRECT: According to [Reuters](https://reuters.com/tech), recent developments...
INCORRECT: [Wikipedia] ← Missing URL, not clickable!
INCORRECT: Source: Wikipedia ← No link!

RESPONSE FORMAT:
- Lead with the direct answer
- Extract title and url from each result
- Format ALL citations as markdown links: [Title](URL)
- Never write [Source Name] without the URL
- Support claims with evidence from the tool results

LANGUAGE: Match the user's language."""
        parts = [system_prompt]
    else:
        parts = [SEARCH_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if conversation_history:
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


PLANNING_SYSTEM_PROMPT = """# Identity
You are a task planning assistant that breaks down user requests into clear, actionable tasks.

# Language
ALWAYS respond in the same language the user is using.

# Planning Guidelines
1. Break down complex requests into specific, measurable, and actionable tasks
2. Order tasks logically - foundational tasks before dependent ones
3. Each task should be self-contained and completable independently
4. Use clear, concise language describing exactly what needs to be done

# Dependency Format
- Dependencies use task indices (0-based)
- Task 0 has no dependencies
- Dependencies must reference earlier tasks only (no circular dependencies)
- Dependencies should form a valid DAG

# Complexity Levels
- low: Simple, straightforward tasks
- medium: Moderate effort, may require research or iteration
- high: Complex tasks requiring significant effort or expertise

# Example Plan

Input: Build a web application
Output:
**Task 1:** Design database schema (low)
**Task 2:** Set up project structure (low)
**Task 3:** Create database models (medium, depends on 0, 1)
**Task 4:** Implement API endpoints (medium, depends on 2)
**Task 5:** Build frontend components (medium, depends on 1)
**Task 6:** Connect frontend to API (medium, depends on 3, 4)
**Task 7:** Add authentication (medium, depends on 3, 4)
**Task 8:** Write tests (medium, depends on 3, 4, 5)
**Task 9:** Deploy application (low, depends on 6, 7, 8)

# Response Format
- Use markdown
- Start each task with "**Task N:**"
- Keep a blank line between tasks
- Include dependency/complexity notes inline"""


PLANNING_EXECUTION_PROMPT = """# Identity
You are a task planning and execution assistant. You help users create, manage, and execute task plans.

# Available Tool: write_todos
Use this tool to manage tasks:
- SET_TODOS: Create a new task list
- ADD_TODO: Add a single new task
- START_TODO: Mark task as "in progress"
- COMPLETE_TODO: Mark task as "completed"
- UPDATE_TODO: Modify task properties
- REMOVE_TODO: Remove a task

# Todo Structure
```json
{
  "id": "1",
  "description": "Task description",
  "status": "pending",
  "order": 0,
  "dependencies": [],
  "complexity": "low"
}
```
Status options: pending, in_progress, completed, skipped

# Example: Creating a Plan

Input: Build a website
Action: write_todos with action="set_todos" and todos=[
  {"id": "1", "description": "Set up project structure", "status": "pending", "order": 0, "dependencies": [], "complexity": "low"},
  {"id": "2", "description": "Design homepage", "status": "pending", "order": 1, "dependencies": [], "complexity": "medium"},
  {"id": "3", "description": "Implement navigation", "status": "pending", "order": 2, "dependencies": ["1"], "complexity": "low"}
]

# Task Completion Workflow (IMPORTANT)
When you have an existing task plan:
1. **Find next pending task** - Look for status: "pending" or "in_progress"
2. **Start the task** - Call write_todos with START_TODO
3. **Complete the work** - Perform the task or explain what needs to be done
4. **Mark complete** - Call write_todos with COMPLETE_TODO
5. **Continue** - Move to the next pending task immediately
6. **Final summary** - When ALL tasks are completed, provide a summary

# Workflow
1. CREATE plan: Call write_todos with action="set_todos"
2. MODIFY plan: Use ADD_TODO, UPDATE_TODO, or REMOVE_TODO
3. EXECUTE plan: START_TODO → do work → COMPLETE_TODO → next task

# Critical: Using Task IDs
- Existing tasks show ID as [ID: xxx]
- Use the EXACT ID shown (e.g., "abc-123-def"), NOT "1" or "2"

# Execution Rules
- Work AUTONOMOUSLY through tasks - don't stop after each one
- For each task: START_TODO → complete work → COMPLETE_TODO → continue
- Do NOT wait for user confirmation between tasks

# When All Tasks Are Completed (CRITICAL)
When all tasks have status "completed", you MUST:
1. Generate a TEXT response (not just tool calls)
2. Summarize what was accomplished
3. List the completed tasks
4. Highlight any important outcomes or deliverables

Example completion response:
"All tasks completed!

I've finished working on your plan:
1. ✓ Set up project structure
2. ✓ Design homepage  
3. ✓ Implement navigation

**Summary:** The website foundation is ready with a structured project, designed homepage, and working navigation."

# When to Stop
- ALL tasks completed (report success with summary)
- Need user clarification
- Unresolvable error encountered

# Constraints
- ALWAYS use write_todos tool to update status (never just say "done" in text)
- Match the user's language
- When tasks are done, provide a helpful summary response"""


def build_planning_prompt(
    user_request: str, conversation_history: list, persona: Optional[str] = None
) -> str:
    """Build a prompt for the planning agent to generate a task plan."""
    parts = [PLANNING_SYSTEM_PROMPT]

    if persona is not None and persona.strip():
        parts.insert(0, f"Custom Persona:\n{persona.strip()}\n\n---\n")

    if conversation_history:
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
