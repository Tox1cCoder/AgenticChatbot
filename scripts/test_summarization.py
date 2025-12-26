"""
Test script for Summarization Middleware.

This script tests the summarization middleware by:
1. Testing the threshold detection logic
2. Testing the summarization with mock messages
3. Running an integration test with lower thresholds

Usage:
    python scripts/test_summarization.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Configure logging to see summarization output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_messages(count: int) -> list:
    """Create mock conversation messages for testing."""
    messages = [SystemMessage(content="You are a helpful assistant.")]
    
    for i in range(count):
        messages.append(HumanMessage(content=f"This is user message number {i + 1}. Can you help me with something?"))
        messages.append(AIMessage(content=f"Of course! I'm happy to help with your request number {i + 1}. What would you like to know?"))
    
    return messages


async def test_threshold_detection():
    """Test that summarization only triggers when thresholds are exceeded."""
    from app.ai.summarization_middleware import (
        summarize_messages_if_needed,
        SummarizationConfig,
        _should_summarize,
        _estimate_tokens,
    )
    
    print("\n" + "=" * 60)
    print("TEST 1: Threshold Detection")
    print("=" * 60)
    
    # Use low thresholds for testing
    config = SummarizationConfig(
        trigger_tokens=100,  # Very low for testing
        trigger_messages=5,  # Very low for testing
        keep_messages=2,
    )
    
    # Test with few messages (should NOT trigger)
    few_messages = create_mock_messages(2)  # 4 messages + 1 system = 5
    print(f"\nTest with {len(few_messages)} messages:")
    print(f"  Estimated tokens: {_estimate_tokens(few_messages)}")
    print(f"  Should summarize: {_should_summarize(few_messages, config)}")
    
    # Test with many messages (should trigger)
    many_messages = create_mock_messages(10)  # 20 messages + 1 system = 21
    print(f"\nTest with {len(many_messages)} messages:")
    print(f"  Estimated tokens: {_estimate_tokens(many_messages)}")
    print(f"  Should summarize: {_should_summarize(many_messages, config)}")
    
    print("\n✅ Threshold detection test completed!")


async def test_summarization_output():
    """Test the actual summarization process."""
    from app.ai.summarization_middleware import (
        summarize_messages_if_needed,
        SummarizationConfig,
    )
    
    print("\n" + "=" * 60)
    print("TEST 2: Summarization Output")
    print("=" * 60)
    
    # Create a conversation with 10 turns (20 messages + system)
    messages = create_mock_messages(10)
    print(f"\nOriginal message count: {len(messages)}")
    
    # Use low thresholds to force summarization
    config = SummarizationConfig(
        trigger_tokens=100,
        trigger_messages=5,
        keep_messages=4,
        model="gemini-3-flash-preview",
    )
    
    print("Calling summarize_messages_if_needed()...")
    result = await summarize_messages_if_needed(messages, config)
    
    print(f"\nResult message count: {len(result)}")
    print(f"Reduction: {len(messages)} -> {len(result)} messages")
    
    # Show the summary message
    for msg in result:
        if isinstance(msg, SystemMessage) and "[Summary" in msg.content:
            print(f"\n📝 Generated Summary:\n{'-' * 40}")
            print(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)
            print("-" * 40)
            break
    
    print("\n✅ Summarization output test completed!")


async def test_no_summarization_when_disabled():
    """Test that summarization doesn't run when disabled."""
    from app.ai.summarization_middleware import (
        summarize_messages_if_needed,
        SummarizationConfig,
    )
    from app.core.config import settings
    
    print("\n" + "=" * 60)
    print("TEST 3: Summarization Disabled")
    print("=" * 60)
    
    # Temporarily disable summarization
    original_value = getattr(settings, "enable_summarization", True)
    settings.enable_summarization = False
    
    try:
        messages = create_mock_messages(10)
        config = SummarizationConfig(
            trigger_tokens=100,
            trigger_messages=5,
            keep_messages=4,
        )
        
        result = await summarize_messages_if_needed(messages, config)
        
        # Should return original messages unchanged
        assert len(result) == len(messages), "Messages should not be modified when disabled"
        print("✅ Summarization correctly skipped when disabled!")
    finally:
        settings.enable_summarization = original_value


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SUMMARIZATION MIDDLEWARE TEST SUITE")
    print("=" * 60)
    
    try:
        await test_threshold_detection()
        await test_summarization_output()
        await test_no_summarization_when_disabled()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
