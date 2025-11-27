import json
import logging
import re
import base64
from typing import Optional, List, Dict, Any, AsyncIterator
from uuid import UUID
from pathlib import Path

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from ..schemas import AgentMessage, AgentResponse, AgentType, MessageRole
from ..prompts import build_rag_prompt
from ...core.config import Settings
from ..mcp_integration import MCPManager
from ...core.config import settings
from ..utils import (
    coerce_response_text,
    extract_agent_execution_info,
    get_error_recovery_hint,
)
from ...repositories.document_image import DocumentImageRepository
from ...database.session import SessionLocal

logger = logging.getLogger(__name__)


class RAGAgent:

    def __init__(
        self,
        settings: Settings,
        qdrant_client: QdrantClient,
        embedding_model: SentenceTransformer,
        collection_name: str = "documents_gemma",
    ):
        self.settings = settings
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

        self.collection_name = collection_name
        self.embedding_dimension = settings.embedding_dimension
        self.model_name = "gemini-3-pro-preview" # "gemini-flash-latest"
        self.gemini_client = None
        self.langchain_model = None
        self.mcp_manager = None
        self.tools = []

        # Store retrieval parameters
        self.top_k = settings.rag_top_k
        self.score_threshold = settings.rag_score_threshold
        self.enable_reranking = settings.enable_reranking
        self.reranker = None

        self._init_gemini()

        # Initialize re-ranker if enabled
        if self.enable_reranking:
            self._init_reranker()

    def _init_gemini(self):
        api_key = self.settings.gemini_api_key
        if not api_key:
            logger.error("Gemini API key not configured")
            return

        if api_key.startswith("GEMINI_API_KEY="):
            api_key = api_key.split("=", 1)[-1].strip()

        self.gemini_client = genai.Client(api_key=api_key)

        self.langchain_model = ChatGoogleGenerativeAI(
            model=self.model_name, google_api_key=api_key, temperature=0.24
        )

    def _init_reranker(self):
        """Initialize the re-ranker model"""
        self.reranker = CrossEncoder(self.settings.reranker_model)
        logger.info(f"Re-ranker initialized: {self.settings.reranker_model}")

    async def _init_tools(self):
        """Initialize MCP manager and load tools useful for document analysis"""
        if self.mcp_manager is None:
            try:
                self.mcp_manager = MCPManager()
                await self.mcp_manager.initialize()
            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP manager for RAGAgent: {e}",
                    exc_info=True,
                )
                self.tools = []
                return

        try:
            all_tools = await self.mcp_manager.get_tools()
        except Exception as e:
            logger.error(
                "Failed to load MCP tools for RAGAgent: %s", e, exc_info=True
            )
            self.tools = []
            return

        self.tools = self._deduplicate_tools(all_tools)

        server_status = self.mcp_manager.get_servers_status()
        active_servers = [
            name for name, status in server_status.items() if status.get("enabled")
        ]
        if self.tools:
            logger.info(
                "Loaded %d MCP tools for RAGAgent from %d servers",
                len(self.tools),
                len(active_servers),
            )
        else:
            logger.warning("No MCP tools available for RAGAgent; running without tools")

    def _deduplicate_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Ensure we only keep one instance of each tool by name."""
        unique_tools: Dict[str, BaseTool] = {}
        for tool in tools or []:
            unique_tools.setdefault(tool.name, tool)
        return list(unique_tools.values())

    def _create_agent_executor(self, tools: List[BaseTool], system_prompt: str):
        """Create agent executor with proper tool binding configuration."""

        tool_choice = (
            settings.tool_choice_mode
            if hasattr(settings, "tool_choice_mode")
            else "auto"
        )

        # Configure model with tool binding
        llm_with_tools = self.langchain_model.bind_tools(
            tools,
            tool_config={
                "function_calling_config": {
                    "mode": (
                        tool_choice.upper()
                        if tool_choice in ["auto", "any", "none"]
                        else "AUTO"
                    )
                }
            },
        )

        agent = create_agent(
            model=llm_with_tools, tools=tools, system_prompt=system_prompt
        )

        return agent

    async def process_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AgentResponse:

        query = message.content
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Initialize tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        retrieved_docs = await self._search(query, conversation_id=conversation_id)

        # Create document grouping structure
        doc_grouping = {}
        doc_id_to_num = {}
        next_doc_num = 1

        for doc in retrieved_docs:
            # Use document_id as primary key, fallback to source
            doc_key = doc.get("document_id") or doc.get("source", "unknown")

            if doc_key not in doc_grouping:
                doc_grouping[doc_key] = {
                    "document_id": doc.get("document_id"),
                    "source": doc.get("source", "unknown"),
                    "document_number": next_doc_num,
                    "chunks": [],
                }
                doc_id_to_num[doc_key] = next_doc_num
                next_doc_num += 1

            # Add chunk details to document group
            chunk_details = {
                "chunk_index": doc.get("chunk_index", 0),
                "score": doc.get("score", 0.0),
                "character_count": len(doc.get("content", "")),
            }
            doc_grouping[doc_key]["chunks"].append(chunk_details)

        has_images = any(doc.get("image_ids") for doc in retrieved_docs)
        images = []
        if has_images:
            images = await self._fetch_images_for_chunks(retrieved_docs)

        prompt = build_rag_prompt(
            query,
            retrieved_docs,
            conversation_history,
            persona=persona,
            document_grouping=doc_grouping,
        )

        response_text: str = ""
        tools_used: List[str] = []
        tool_artifacts: List[Dict[str, Any]] = []
        error_message: Optional[str] = None

        try:
            if images and self.tools and self.langchain_model:
                (
                    tool_response_text,
                    tools_used,
                    tool_artifacts,
                ) = await self._generate_with_tools(prompt)

                multimodal_prompt = self._augment_prompt_with_tool_context(
                    prompt, tool_response_text, tool_artifacts
                )
                response_text = await self._generate_with_vision(
                    multimodal_prompt, images
                )
            elif images:
                response_text = await self._generate_with_vision(prompt, images)
            elif self.tools and self.langchain_model:
                # Use tools without images
                response_text, tools_used, tool_artifacts = (
                    await self._generate_with_tools(prompt)
                )
            else:
                # Regular text-only generation
                response_text = await self._generate(prompt)
        except Exception as exc:
            logger.error("Error generating RAG response: %s", exc, exc_info=True)
            error_message = f"{type(exc).__name__}: {exc}"
            tools_used = []
            tool_artifacts = [
                {
                    "tool": "rag_agent",
                    "args": {},
                    "error": error_message,
                    "hint": get_error_recovery_hint(exc, "rag_agent", {}),
                }
            ]

        response_text = coerce_response_text(response_text)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=response_text
        )

        # Build grouped citations structure (documents_cited)
        documents_cited = []
        for doc_key, doc_info in doc_grouping.items():
            # Calculate aggregate stats for this document
            chunks = doc_info["chunks"]
            total_chunks = len(chunks)
            avg_score = (
                sum(c["score"] for c in chunks) / total_chunks
                if total_chunks > 0
                else 0.0
            )

            document_entry = {
                "document_id": doc_info["document_id"],
                "source": doc_info["source"],
                "document_number": doc_info["document_number"],
                "chunks": chunks,
                "total_chunks": total_chunks,
                "avg_score": avg_score,
            }
            documents_cited.append(document_entry)

        # Sort by document number for consistency
        documents_cited.sort(key=lambda x: x["document_number"])

        # Build legacy flat citations for backward compatibility
        all_citations = [
            {
                "source": doc.get("source", "unknown"),
                "score": doc.get("score", 0.0),
                "chunk_index": doc.get("chunk_index", 0),
                "character_count": len(doc.get("content", "")),
            }
            for doc in retrieved_docs
        ]

        # Apply citation verification if enabled
        citations = all_citations
        citation_verification_enabled = False
        citation_coverage = 100.0

        if self.settings.enable_citation_verification and retrieved_docs:
            verified_citations = self._verify_citations(
                response_text, retrieved_docs, all_citations, doc_grouping
            )
            if verified_citations is not None:
                citations = verified_citations
                citation_verification_enabled = True
                citation_coverage = (
                    (len(citations) / len(all_citations) * 100)
                    if all_citations
                    else 0.0
                )

        # Calculate retrieval statistics
        avg_score = (
            sum(doc.get("score", 0.0) for doc in retrieved_docs) / len(retrieved_docs)
            if retrieved_docs
            else 0.0
        )

        # Build metadata
        metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "documents_found": len(doc_grouping),  # Number of unique documents
            "chunks_retrieved": len(retrieved_docs),  # Total number of chunks
            "documents_cited": documents_cited,
            "citations": citations,
            "context_messages": len(conversation_history),
            "retrieval_stats": {
                "total_retrieved": len(retrieved_docs),
                "avg_score": avg_score,
            },
            "persona_used": persona,
            "tools_available": len(self.tools),
            "citation_verification_enabled": citation_verification_enabled,
            "citation_coverage": citation_coverage,
            "has_images": bool(images),
            "images_count": len(images) if images else 0,
        }

        # Add tool usage metadata if tools were used
        if tools_used:
            metadata["tools_used"] = tools_used
            metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
            if not error_message:
                error_entries = [
                    artifact.get("error")
                    for artifact in tool_artifacts
                    if artifact.get("error")
                ]
                if error_entries:
                    error_message = error_entries[0]
                    metadata["error"] = error_message
        elif error_message:
            metadata["error"] = error_message

        return AgentResponse(
            agent_type=AgentType.RAG,
            agent_id="rag_agent",
            message=response_message,
            metadata=metadata,
            tool_artifacts=tool_artifacts if tool_artifacts else None,
            error=error_message,
        )

    async def stream_message(
        self,
        message: AgentMessage,
        conversation_id: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream message processing with token-by-token generation.
        Yields events as tokens are generated.
        """
        query = message.content
        conversation_history = message.metadata.get("history", [])
        persona = message.metadata.get("persona")

        # Initialize tools if not done yet
        if self.mcp_manager is None:
            await self._init_tools()

        retrieved_docs = await self._search(query, conversation_id=conversation_id)

        # Create document grouping structure
        doc_grouping = {}
        doc_id_to_num = {}
        next_doc_num = 1

        for doc in retrieved_docs:
            doc_key = doc.get("document_id") or doc.get("source", "unknown")

            if doc_key not in doc_grouping:
                doc_grouping[doc_key] = {
                    "document_id": doc.get("document_id"),
                    "source": doc.get("source", "unknown"),
                    "document_number": next_doc_num,
                    "chunks": [],
                }
                doc_id_to_num[doc_key] = next_doc_num
                next_doc_num += 1

            chunk_details = {
                "chunk_index": doc.get("chunk_index", 0),
                "score": doc.get("score", 0.0),
                "character_count": len(doc.get("content", "")),
            }
            doc_grouping[doc_key]["chunks"].append(chunk_details)

        has_images = any(doc.get("image_ids") for doc in retrieved_docs)
        images = []
        if has_images:
            images = await self._fetch_images_for_chunks(retrieved_docs)

        prompt = build_rag_prompt(
            query,
            retrieved_docs,
            conversation_history,
            persona=persona,
            document_grouping=doc_grouping,
        )

        accumulated_content = ""
        tools_used: List[str] = []
        tool_artifacts: List[Dict[str, Any]] = []
        error_message: Optional[str] = None

        try:
            if images and self.tools and self.langchain_model:
                # Complex case: images + tools - use non-streaming fallback
                (
                    tool_response_text,
                    tools_used,
                    tool_artifacts,
                ) = await self._generate_with_tools(prompt)

                multimodal_prompt = self._augment_prompt_with_tool_context(
                    prompt, tool_response_text, tool_artifacts
                )
                response_text = await self._generate_with_vision(
                    multimodal_prompt, images
                )
                # Yield as single token
                yield {"type": "token", "content": response_text}
                accumulated_content = response_text
            elif images:
                # Vision only - non-streaming
                response_text = await self._generate_with_vision(prompt, images)
                yield {"type": "token", "content": response_text}
                accumulated_content = response_text
            elif self.tools and self.langchain_model:
                # Tools only - stream
                async for event in self._generate_with_tools_stream(prompt):
                    if event["type"] == "token":
                        accumulated_content += event["content"]
                        yield event
                    elif event["type"] in ["tool_start", "tool_end"]:
                        yield event
                    elif event["type"] == "result":
                        accumulated_content = event["response_text"]
                        tools_used = event["tools_used"]
                        tool_artifacts = event["tool_artifacts"]
            else:
                # Text only - stream
                async for chunk in self._generate_stream(prompt):
                    accumulated_content += chunk
                    yield {"type": "token", "content": chunk}

        except Exception as exc:
            logger.error("Error streaming RAG response: %s", exc, exc_info=True)
            error_message = f"{type(exc).__name__}: {exc}"
            tools_used = []
            tool_artifacts = [
                {
                    "tool": "rag_agent",
                    "args": {},
                    "error": error_message,
                    "hint": get_error_recovery_hint(exc, "rag_agent", {}),
                }
            ]

        accumulated_content = coerce_response_text(accumulated_content)

        response_message = AgentMessage(
            role=MessageRole.ASSISTANT, content=accumulated_content
        )

        # Build grouped citations structure (documents_cited)
        documents_cited = []
        for doc_key, doc_info in doc_grouping.items():
            chunks = doc_info["chunks"]
            total_chunks = len(chunks)
            avg_score = (
                sum(c["score"] for c in chunks) / total_chunks
                if total_chunks > 0
                else 0.0
            )

            document_entry = {
                "document_id": doc_info["document_id"],
                "source": doc_info["source"],
                "document_number": doc_info["document_number"],
                "chunks": chunks,
                "total_chunks": total_chunks,
                "avg_score": avg_score,
            }
            documents_cited.append(document_entry)

        documents_cited.sort(key=lambda x: x["document_number"])

        # Build legacy flat citations
        all_citations = [
            {
                "source": doc.get("source", "unknown"),
                "score": doc.get("score", 0.0),
                "chunk_index": doc.get("chunk_index", 0),
                "character_count": len(doc.get("content", "")),
            }
            for doc in retrieved_docs
        ]

        citations = all_citations
        citation_verification_enabled = False
        citation_coverage = 100.0

        if self.settings.enable_citation_verification and retrieved_docs:
            verified_citations = self._verify_citations(
                accumulated_content, retrieved_docs, all_citations, doc_grouping
            )
            if verified_citations is not None:
                citations = verified_citations
                citation_verification_enabled = True
                citation_coverage = (
                    (len(citations) / len(all_citations) * 100)
                    if all_citations
                    else 0.0
                )

        avg_score = (
            sum(doc.get("score", 0.0) for doc in retrieved_docs) / len(retrieved_docs)
            if retrieved_docs
            else 0.0
        )

        # Build metadata
        metadata = {
            "model": self.model_name,
            "conversation_id": conversation_id,
            "documents_found": len(doc_grouping),
            "chunks_retrieved": len(retrieved_docs),
            "documents_cited": documents_cited,
            "citations": citations,
            "context_messages": len(conversation_history),
            "retrieval_stats": {
                "total_retrieved": len(retrieved_docs),
                "avg_score": avg_score,
            },
            "persona_used": persona,
            "tools_available": len(self.tools),
            "citation_verification_enabled": citation_verification_enabled,
            "citation_coverage": citation_coverage,
            "has_images": bool(images),
            "images_count": len(images) if images else 0,
        }

        if tools_used:
            metadata["tools_used"] = tools_used
            metadata["tool_calls_count"] = len(tools_used)
        if tool_artifacts:
            metadata["tool_artifacts"] = tool_artifacts
            if not error_message:
                error_entries = [
                    artifact.get("error")
                    for artifact in tool_artifacts
                    if artifact.get("error")
                ]
                if error_entries:
                    error_message = error_entries[0]
                    metadata["error"] = error_message
        elif error_message:
            metadata["error"] = error_message

        # Yield complete event
        yield {
            "type": "complete",
            "response": AgentResponse(
                agent_type=AgentType.RAG,
                agent_id="rag_agent",
                message=response_message,
                metadata=metadata,
                tool_artifacts=tool_artifacts if tool_artifacts else None,
                error=error_message,
            ),
        }

    async def _generate_with_tools_stream(
        self, prompt: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate streaming response with tool calling support.
        Yields token chunks and tool execution events.
        """
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, prompt)

            accumulated_text = ""

            # Stream agent execution
            async for event in agent_executor.astream_events(
                {"messages": [HumanMessage(content=prompt)]}, version="v1"
            ):
                event_type = event.get("event")

                # Extract tokens from LLM events
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        # Handle both string and list content
                        token = chunk.content
                        if isinstance(token, list):
                            # If it's a list, join the string parts
                            token = "".join(str(item) for item in token if item)
                        if token:  # Only yield non-empty tokens
                            accumulated_text += token
                            yield {"type": "token", "content": token}

                # Track tool execution
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown_tool")
                    yield {"type": "tool_start", "name": tool_name}

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown_tool")
                    yield {"type": "tool_end", "name": tool_name}

            # Get final response for complete extraction
            agent_response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            # Yield result info
            yield {
                "type": "result",
                "response_text": response_text,
                "tools_used": tools_used,
                "tool_artifacts": tool_artifacts,
            }

        except Exception as exc:
            logger.error("Error in RAGAgent tool streaming: %s", exc, exc_info=True)
            raise

    def _verify_citations(
        self,
        response_text: str,
        retrieved_docs: List[Dict[str, Any]],
        all_citations: List[Dict[str, Any]],
        doc_grouping: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        try:
            citation_patterns = [
                r"\[Document\s+(\d+)\]",  # [Document 1]
                r"Document\s+(\d+)",  # Document 1
                r"\[(\d+)\]",  # [1]
                r"\(Document\s+(\d+)\)",  # (Document 1)
                r"\((\d+)\)",  # (1)
            ]

            referenced_doc_numbers = set()

            # Try each pattern to find document references
            for pattern in citation_patterns:
                matches = re.finditer(pattern, response_text, re.IGNORECASE)
                for match in matches:
                    try:
                        doc_num = int(match.group(1))
                        # Document numbers are 1-indexed in text
                        if 1 <= doc_num <= len(doc_grouping):
                            referenced_doc_numbers.add(doc_num)
                    except (ValueError, IndexError):
                        continue

            if not referenced_doc_numbers:
                if self.settings.min_citation_coverage > 0:
                    for citation in all_citations:
                        citation["potentially_relevant"] = True
                    return all_citations
                else:
                    return []

            # Filter citations to only include chunks from referenced documents
            verified_citations = []
            for citation in all_citations:
                # Find which document this chunk belongs to
                source = citation.get("source", "unknown")
                for doc_key, doc_info in doc_grouping.items():
                    if doc_info["source"] == source:
                        if doc_info["document_number"] in referenced_doc_numbers:
                            citation_copy = citation.copy()
                            citation_copy["referenced"] = True
                            verified_citations.append(citation_copy)
                        break

            return verified_citations

        except Exception as exc:
            logger.error(f"Error in citation verification: {exc}", exc_info=True)
            # On error, return all citations to avoid losing information
            return all_citations

    async def _generate_stream(self, prompt: str):
        try:
            response_stream = self.gemini_client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
            )

            for chunk in response_stream:
                if hasattr(chunk, "text"):
                    yield chunk.text

        except Exception as exc:
            logger.error(f"Error in streaming generation: {exc}", exc_info=True)
            raise RuntimeError(f"Gemini streaming API error: {exc}") from exc

    async def _generate_with_tools(
        self, prompt: str, streaming_callback=None
    ) -> tuple[str, List[str], List[Dict[str, Any]]]:
        """Generate response with tool calling support using create_agent."""
        try:
            # Create agent executor
            agent_executor = self._create_agent_executor(self.tools, prompt)

            # Invoke agent with the user message
            agent_response = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=prompt)]}
            )

            # Extract execution info
            execution_info = extract_agent_execution_info(agent_response)

            response_text = execution_info["response_text"]
            tools_used = execution_info["tools_used"]
            tool_artifacts = execution_info["tool_artifacts"]

            return response_text, tools_used, tool_artifacts

        except Exception as exc:
            logger.error("Error in RAGAgent tool calling flow: %s", exc, exc_info=True)
            raise

    def _augment_prompt_with_tool_context(
        self,
        base_prompt: str,
        tool_response_text: str,
        tool_artifacts: List[Dict[str, Any]],
    ) -> str:
        """Combine tool outputs with the base prompt for multimodal generation."""
        sections: List[str] = [base_prompt.rstrip()]

        context_chunks: List[str] = []

        if tool_response_text:
            context_chunks.append(
                "Tool-assisted analysis:\n" + tool_response_text.strip()
            )

        if tool_artifacts:
            try:
                artifacts_dump = json.dumps(tool_artifacts, indent=2, default=str)
            except TypeError:
                artifacts_dump = str(tool_artifacts)
            context_chunks.append(f"Tool call details:\n{artifacts_dump}")

        if context_chunks:
            sections.append(
                "-----\nLeverage the following tool outputs alongside the attached document images:\n"
                + "\n\n".join(context_chunks)
            )

        sections.append(
            "When responding, cite document evidence and reference images by index where relevant."
        )

        return "\n\n".join(sections)

    async def _search(
        self, query: str, top_k: int = None, conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.top_k

        query_embedding = self.embedding_model.encode(query).tolist()

        search_filter = None

        if conversation_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id", match=MatchValue(value=conversation_id)
                    )
                ]
            )

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=self.score_threshold,
            query_filter=search_filter,
        )

        if not search_results and conversation_id:
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=self.score_threshold,
            )

        results = []
        for result in search_results:
            results.append(
                {
                    "content": result.payload.get("content", ""),
                    "source": result.payload.get("source", "unknown"),
                    "score": result.score,
                    "page_number": result.payload.get("page_number"),
                    "page_start": result.payload.get("page_start"),
                    "page_end": result.payload.get("page_end"),
                    "document_id": result.payload.get("document_id", ""),
                    "conversation_id": result.payload.get("conversation_id", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "has_tables": result.payload.get("has_tables", False),
                    "table_count": result.payload.get("table_count", 0),
                    "image_ids": result.payload.get("image_ids", []),
                    "image_paths": result.payload.get("image_paths", []),
                    "image_captions": result.payload.get("image_captions", []),
                }
            )

        if self.enable_reranking and len(results) > 3:
            results = await self._rerank_results(query, results)

        return results

    async def _rerank_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank search results using cross-encoder"""
        if not self.reranker or not results:
            return results

        # Prepare pairs for re-ranking
        pairs = [[query, doc["content"]] for doc in results]

        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)

        # Add rerank scores to results
        for i, score in enumerate(rerank_scores):
            results[i]["rerank_score"] = float(score)

        # Sort by rerank score
        results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Keep only top K after re-ranking
        results = results[: self.settings.rerank_top_k]

        return results

    async def _fetch_images_for_chunks(
        self, retrieved_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        images = []
        seen_image_ids = set()

        for doc in retrieved_docs:
            image_ids = doc.get("image_ids", [])
            if not image_ids:
                continue

            for image_id in image_ids:
                if image_id and image_id not in seen_image_ids:
                    seen_image_ids.add(image_id)

        if not seen_image_ids:
            return images

        image_repo = DocumentImageRepository(SessionLocal)

        for image_id in seen_image_ids:
            try:
                image_uuid = UUID(str(image_id))
            except Exception:
                continue

            image = image_repo.get_by_id(image_uuid)
            if not image:
                continue

            image_path = Path(image.image_path)
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path

            if not image_path.exists():
                continue

            with open(image_path, "rb") as f:
                image_bytes = f.read()

            base64_data = base64.b64encode(image_bytes).decode("utf-8")

            images.append(
                {
                    "id": str(image.id),
                    "data": base64_data,
                    "mime_type": image.mime_type,
                    "caption": image.image_caption,
                    "page_number": image.page_number,
                    "source_path": str(image_path),
                }
            )

        return images

    async def _generate_with_vision(
        self, prompt: str, images: List[Dict[str, Any]]
    ) -> str:
        try:
            parts = []

            parts.append(types.Part(text=prompt))

            for index, image in enumerate(images, start=1):
                image_data = base64.b64decode(image["data"])

                mime_type = (image.get("mime_type") or "image/jpeg").strip()
                if mime_type.lower() == "image/jpg":
                    mime_type = "image/jpeg"
                parts.append(
                    types.Part.from_bytes(data=image_data, mime_type=mime_type)
                )

            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=parts
            )

            return response.text if hasattr(response, "text") else str(response)

        except Exception as exc:
            logger.error(f"Error in vision generation: {exc}", exc_info=True)
            raise

    async def _generate(self, prompt: str) -> str:
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    async def initialize(self):
        """Initialize the RAG agent"""
        return True

    async def cleanup(self):
        """Cleanup resources"""
        # Cleanup MCP resources
        if self.mcp_manager:
            await self.mcp_manager.cleanup()

        # Cleanup Qdrant
        if hasattr(self.qdrant_client, "close"):
            self.qdrant_client.close()

    def get_status(self) -> dict:
        """Get the current status of the RAG agent"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            collection_info = None
            if collection_exists:
                collection_info = self.qdrant_client.get_collection(
                    self.collection_name
                )

            return {
                "status": "healthy",
                "collection_exists": collection_exists,
                "collection_name": self.collection_name,
                "vectors_count": (
                    collection_info.vectors_count if collection_info else 0
                ),
                "embedding_model": self.embedding_model,
                "embedding_dimension": self.embedding_dimension,
            }
        except Exception as e:
            logger.error(f"Error getting RAG agent status: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def delete_document_vectors(self, document_id: str) -> dict:
        """
        Delete all vectors associated with a document ID and cleanup associated images.
        """
        try:
            # Delete associated images from database and filesystem
            images_deleted = 0

            image_repo = DocumentImageRepository(SessionLocal)

            # Get image paths before deletion
            image_paths = image_repo.get_image_paths_by_document_id(UUID(document_id))

            # Delete from database
            images_deleted = image_repo.delete_by_document_id(UUID(document_id))

            # Delete image files from filesystem
            for image_path in image_paths:
                # Resolve relative paths to absolute paths
                path_obj = Path(image_path)
                if not path_obj.is_absolute():
                    path_obj = Path.cwd() / path_obj

                if path_obj.exists():
                    path_obj.unlink()

            # Delete document image folder if empty
            doc_image_folder = Path(settings.document_images_storage_path) / document_id
            if doc_image_folder.exists() and not any(doc_image_folder.iterdir()):
                doc_image_folder.rmdir()

            # Delete vectors from Qdrant
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            )

            result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=delete_filter),
            )

            return {
                "success": True,
                "document_id": document_id,
                "images_deleted": images_deleted,
                "message": f"Vectors and {images_deleted} images deleted for document {document_id}",
                "operation_result": str(result),
            }
        except Exception as e:
            logger.error(
                f"Error deleting vectors for document {document_id}: {e}", exc_info=True
            )
            return {"success": False, "document_id": document_id, "error": str(e)}
