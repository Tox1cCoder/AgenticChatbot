"""
Agent Router Implementation

This module implements the intelligent routing system that determines which agent
should handle a specific request based on content analysis and agent capabilities.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
from uuid import UUID
import asyncio
import logging
import re
from datetime import datetime

from app.ai.interfaces import IAgentRouter, IAgent
from app.ai.schemas import (
    AgentRequest,
    AgentType,
    AgentCapability,
    RoutingStrategy,
    RouterConfig,
    AgentError,
    AgentRegistration,
    AgentRegistry,
)


class ContentAnalyzer:
    """
    Analyzes request content to determine appropriate routing.
    """
    
    # Keywords that suggest different agent types
    CHAT_KEYWORDS = {
        "hello", "hi", "how are you", "what's up", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "bye", "goodbye", "see you",
        "tell me about yourself", "who are you", "what can you do"
    }
    
    RAG_KEYWORDS = {
        "search", "find", "lookup", "document", "file", "what is", "explain",
        "definition", "meaning", "how to", "tutorial", "guide", "information",
        "details", "specification", "documentation", "manual", "reference"
    }
    
    QUESTION_PATTERNS = [
        r"^(what|who|when|where|why|how)\s+",
        r"\?$",
        r"^(can|could|would|should|do|does|is|are)\s+",
        r"^(tell me|show me|explain|describe)\s+",
    ]
    
    def __init__(self):
        self.logger = logging.getLogger("router.content_analyzer")
    
    def analyze_content(self, content: str) -> Dict[AgentType, float]:
        """
        Analyze content and return confidence scores for each agent type.
        
        Args:
            content: The content to analyze
            
        Returns:
            Dict[AgentType, float]: Confidence scores for each agent type
        """
        content_lower = content.lower().strip()
        scores = {agent_type: 0.0 for agent_type in AgentType}
        
        # Analyze for chat patterns
        chat_score = self._calculate_chat_score(content_lower)
        scores[AgentType.CHAT] = chat_score
        
        # Analyze for RAG patterns
        rag_score = self._calculate_rag_score(content_lower)
        scores[AgentType.RAG] = rag_score
        
        # Router should never be directly selected by content analysis
        scores[AgentType.ROUTER] = 0.0
        
        # Normalize scores to ensure they sum to 1.0
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            # Default to chat if no clear pattern
            scores[AgentType.CHAT] = 1.0
        
        self.logger.debug(f"Content analysis scores: {scores}")
        return scores
    
    def _calculate_chat_score(self, content: str) -> float:
        """Calculate confidence score for chat agent"""
        score = 0.0
        
        # Check for chat keywords
        for keyword in self.CHAT_KEYWORDS:
            if keyword in content:
                score += 0.3
        
        # Check for conversational patterns
        if any(greeting in content for greeting in ["hello", "hi", "hey"]):
            score += 0.4
        
        if any(polite in content for polite in ["please", "thanks", "thank you"]):
            score += 0.2
        
        # Check for personal questions
        if any(personal in content for personal in ["you", "your", "yourself"]):
            score += 0.1
        
        # Short messages tend to be conversational
        if len(content.split()) <= 5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_rag_score(self, content: str) -> float:
        """Calculate confidence score for RAG agent"""
        score = 0.0
        
        # Check for RAG keywords
        for keyword in self.RAG_KEYWORDS:
            if keyword in content:
                score += 0.3
        
        # Check for question patterns
        for pattern in self.QUESTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                score += 0.4
                break
        
        # Longer, more complex questions tend to need RAG
        word_count = len(content.split())
        if word_count > 10:
            score += 0.2
        elif word_count > 20:
            score += 0.3
        
        # Check for specific information requests
        if any(info_word in content for info_word in ["specific", "exact", "precise", "detailed"]):
            score += 0.2
        
        return min(score, 1.0)


class LoadBalancer:
    """
    Handles load balancing when multiple agents of the same type are available.
    """
    
    def __init__(self):
        self.agent_request_counts: Dict[str, int] = {}
        self.logger = logging.getLogger("router.load_balancer")
    
    def select_agent(self, agents: List[AgentRegistration]) -> Optional[AgentRegistration]:
        """
        Select an agent from available agents using round-robin load balancing.
        
        Args:
            agents: List of available agents
            
        Returns:
            Optional[AgentRegistration]: Selected agent or None if no agents available
        """
        if not agents:
            return None
        
        # Filter out unhealthy agents
        healthy_agents = [agent for agent in agents if agent.status == "active"]
        if not healthy_agents:
            return None
        
        # Use round-robin selection
        if len(healthy_agents) == 1:
            selected = healthy_agents[0]
        else:
            # Find agent with lowest request count
            min_count = float('inf')
            selected = None
            
            for agent in healthy_agents:
                count = self.agent_request_counts.get(agent.instance_id, 0)
                if count < min_count:
                    min_count = count
                    selected = agent
        
        if selected:
            # Increment request count
            self.agent_request_counts[selected.instance_id] = (
                self.agent_request_counts.get(selected.instance_id, 0) + 1
            )
            self.logger.debug(f"Selected agent {selected.instance_id} for load balancing")
        
        return selected


class AgentRouter(IAgentRouter):
    """
    Main router implementation that orchestrates agent selection.
    """
    
    def __init__(self, config: RouterConfig):
        """
        Initialize the agent router.
        
        Args:
            config: Router configuration
        """
        self.config = config
        self.registry = AgentRegistry()
        self.content_analyzer = ContentAnalyzer()
        self.load_balancer = LoadBalancer()
        self.logger = logging.getLogger("agent_router")
        
        # Track routing history for analytics
        self.routing_history: List[Dict] = []
    
    async def route_request(self, request: AgentRequest) -> AgentType:
        """
        Route a request to the most appropriate agent.
        
        Args:
            request: The request to route
            
        Returns:
            AgentType: The type of agent best suited for this request
            
        Raises:
            AgentError: If routing fails
        """
        try:
            start_time = datetime.utcnow()
            
            # If target agent is specified, validate and use it
            if request.target_agent:
                if self._is_agent_available(request.target_agent):
                    self.logger.info(f"Using specified target agent: {request.target_agent}")
                    return request.target_agent
                else:
                    self.logger.warning(f"Specified target agent {request.target_agent} not available")
            
            # Get agent scores based on configured strategy
            agent_scores = await self.get_agent_scores(request)
            
            # Select agent with highest score
            if not agent_scores:
                self.logger.warning("No agent scores available, using fallback")
                selected_agent = self.config.fallback_agent
            else:
                # Find agent with highest score above threshold
                best_agent = None
                best_score = 0.0
                
                for agent_type, score in agent_scores.items():
                    if score > best_score and score >= self.config.confidence_threshold:
                        best_score = score
                        best_agent = agent_type
                
                if best_agent is None:
                    self.logger.warning(
                        f"No agent met confidence threshold {self.config.confidence_threshold}, "
                        f"using fallback"
                    )
                    selected_agent = self.config.fallback_agent
                else:
                    selected_agent = best_agent
            
            # Record routing decision
            routing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._record_routing_decision(request, selected_agent, agent_scores, routing_time)
            
            self.logger.info(
                f"Routed request {request.request_id} to {selected_agent} "
                f"(confidence: {agent_scores.get(selected_agent, 0.0):.2f})"
            )
            
            return selected_agent
            
        except Exception as e:
            self.logger.error(f"Failed to route request {request.request_id}: {e}")
            raise AgentError(
                agent_id="router",
                agent_type=AgentType.ROUTER,
                error_type="routing_error",
                error_message=f"Failed to route request: {str(e)}",
                request_id=request.request_id
            )
    
    async def get_agent_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """
        Get confidence scores for all available agents for a given request.
        
        Args:
            request: The request to evaluate
            
        Returns:
            Dict[AgentType, float]: Mapping of agent types to confidence scores
        """
        if self.config.default_strategy == RoutingStrategy.CONTENT_BASED:
            return self._get_content_based_scores(request)
        elif self.config.default_strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._get_capability_based_scores(request)
        elif self.config.default_strategy == RoutingStrategy.INTENT_BASED:
            return await self._get_intent_based_scores(request)
        else:  # ROUND_ROBIN
            return self._get_round_robin_scores(request)
    
    def register_agent(self, agent: IAgent) -> None:
        """
        Register an agent with the router.
        
        Args:
            agent: The agent to register
        """
        registration = AgentRegistration(
            config=agent.config,
            instance_id=agent.agent_id
        )
        
        self.registry.agents[agent.agent_id] = registration
        self.logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the router.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.registry.agents:
            del self.registry.agents[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
    
    def _get_content_based_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """Get scores based on content analysis"""
        return self.content_analyzer.analyze_content(request.message.content)
    
    def _get_capability_based_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """Get scores based on required capabilities"""
        scores = {agent_type: 0.0 for agent_type in AgentType}
        
        if not request.required_capabilities:
            # No specific capabilities required, use content-based routing
            return self._get_content_based_scores(request)
        
        # Score agents based on capability match
        for agent_type in AgentType:
            available_agents = self.registry.get_agents_by_type(agent_type)
            if not available_agents:
                continue
            
            # Check if any agent of this type has the required capabilities
            max_capability_score = 0.0
            for agent_registration in available_agents:
                agent_capabilities = set(agent_registration.config.capabilities)
                required_capabilities = set(request.required_capabilities)
                
                if required_capabilities.issubset(agent_capabilities):
                    # All required capabilities are available
                    capability_score = 1.0
                else:
                    # Partial match
                    overlap = len(required_capabilities.intersection(agent_capabilities))
                    capability_score = overlap / len(required_capabilities)
                
                max_capability_score = max(max_capability_score, capability_score)
            
            scores[agent_type] = max_capability_score
        
        return scores
    
    async def _get_intent_based_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """Get scores based on intent analysis (placeholder for ML-based intent detection)"""
        # For now, fall back to content-based routing
        # In a full implementation, this would use NLP models to detect intent
        self.logger.debug("Intent-based routing not fully implemented, using content-based")
        return self._get_content_based_scores(request)
    
    def _get_round_robin_scores(self, request: AgentRequest) -> Dict[AgentType, float]:
        """Get scores for round-robin routing"""
        available_types = []
        for agent_type in AgentType:
            if agent_type == AgentType.ROUTER:
                continue
            if self.registry.get_agents_by_type(agent_type):
                available_types.append(agent_type)
        
        if not available_types:
            return {}
        
        # Simple round-robin: rotate through available types
        total_requests = len(self.routing_history)
        selected_index = total_requests % len(available_types)
        selected_type = available_types[selected_index]
        
        scores = {agent_type: 0.0 for agent_type in AgentType}
        scores[selected_type] = 1.0
        
        return scores
    
    def _is_agent_available(self, agent_type: AgentType) -> bool:
        """Check if any agent of the specified type is available"""
        agents = self.registry.get_agents_by_type(agent_type)
        return len(agents) > 0
    
    def _record_routing_decision(
        self,
        request: AgentRequest,
        selected_agent: AgentType,
        scores: Dict[AgentType, float],
        routing_time_ms: float
    ) -> None:
        """Record routing decision for analytics"""
        decision = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(request.request_id),
            "conversation_id": str(request.conversation_id),
            "user_id": str(request.user_id),
            "selected_agent": selected_agent.value,
            "scores": {k.value: v for k, v in scores.items()},
            "routing_time_ms": routing_time_ms,
            "strategy": self.config.default_strategy.value,
            "message_length": len(request.message.content),
            "required_capabilities": [cap.value for cap in request.required_capabilities]
        }
        
        self.routing_history.append(decision)
        
        # Keep only recent history to prevent memory issues
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-500:]
    
    def get_routing_analytics(self) -> Dict:
        """Get analytics about routing decisions"""
        if not self.routing_history:
            return {"total_requests": 0}
        
        total_requests = len(self.routing_history)
        agent_counts = {}
        avg_routing_time = 0.0
        
        for decision in self.routing_history:
            agent = decision["selected_agent"]
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            avg_routing_time += decision["routing_time_ms"]
        
        avg_routing_time /= total_requests
        
        return {
            "total_requests": total_requests,
            "agent_distribution": agent_counts,
            "average_routing_time_ms": avg_routing_time,
            "current_strategy": self.config.default_strategy.value
        }


# Factory function for creating configured router
def create_agent_router(config: Optional[RouterConfig] = None) -> AgentRouter:
    """
    Create a configured agent router.
    
    Args:
        config: Optional router configuration
        
    Returns:
        AgentRouter: Configured router instance
    """
    if config is None:
        config = RouterConfig()
    
    return AgentRouter(config)


__all__ = [
    "ContentAnalyzer",
    "LoadBalancer", 
    "AgentRouter",
    "create_agent_router",
]
