import pytest
from .prompts import EXTRACT_PROMPT
from .test_utils import chat, make_client, semantic_match_multi


@pytest.mark.expensive
def test_extract():
    """Test the extract prompt with real API calls. Marked as expensive since it makes API calls."""
    client = make_client()

    # Test case 1: Basic fact extraction
    message = "I'm John Doe, and I'd like to return the shoes I bought last week."
    result = chat(client, message, EXTRACT_PROMPT)
    expected_facts = [
        "Customer name: John Doe",
        "Wants to return shoes",
        "Purchase made last week"
    ]
    assert semantic_match_multi(expected_facts,
                                result["facts"]), f"Failed on basic facts: {result['facts']}"

    # Test case 2: Belief extraction
    message = "I think that we should build more housing."
    result = chat(client, message, EXTRACT_PROMPT)
    expected_belief = ["User believes that we should build more housing"]
    assert semantic_match_multi(expected_belief,
                                result["facts"]), f"Failed on belief: {result['facts']}"

    # Test case 3: Empty response for greetings
    message = "Hi there, how are you doing today?"
    result = chat(client, message, EXTRACT_PROMPT)
    assert len(result["facts"]) == 0, f"Failed on greeting: {result['facts']}"

    # Test case 4: Multiple facts with contact info
    message = "My name is Jane Smith, I ordered product XYZ-123 on May 15th, and you can reach me at jane@example.com."
    result = chat(client, message, EXTRACT_PROMPT)
    expected_facts = [
        "Customer name: Jane Smith",
        "Ordered product XYZ-123",
        "Order date: May 15th",
        "Email: jane@example.com"
    ]
    assert semantic_match_multi(expected_facts,
                                result["facts"]), f"Failed on multiple facts: {result['facts']}"
