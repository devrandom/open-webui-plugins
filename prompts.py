from test_utils import semantic_match_multi, make_client, chat

EXTRACT_PROMPT = f"""
You will extract objective facts and subjective beliefs from user messages and format them in a structured JSON response.

## Instructions:
1. Extract only clear, concrete facts that are directly stated by the user
2. Identify subjective beliefs, opinions, or preferences and label them as "believes" or "thinks"
3. Skip greetings, small talk, or contextual statements without factual content
4. Be concise but precise in your extractions
5. Format each fact as a distinct item in the JSON array

## Examples:

Input: Hi there, how are you?
Output: {{"facts": []}}

Input: The weather is nice today.
Output: {{"facts": []}}

Input: My order #12345 hasn't arrived yet and I placed it 3 days ago.
Output: {{"facts": ["Order #12345 not received", "Order placed 3 days ago"]}}

Input: I'm John Doe, and I'd like to return the shoes I bought last week. They're too small.
Output: {{"facts": ["Customer name: John Doe", "Wants to return shoes", "Purchase made last week", "Reason for return: shoes too small"]}}

Input: I think we should keep our desks tidy to improve productivity.
Output: {{"facts": ["User believes that we should keep our desks tidy", "User believes tidy desks improve productivity"]}}

Input: I'd prefer to be contacted by email at john@example.com rather than by phone.
Output: {{"facts": ["Email address: john@example.com", "User prefers email contact over phone contact"]}}

Always respond with valid JSON containing only the "facts" array.
"""


def test_extract():
    client = make_client()
    
    # Test case 1: Basic fact extraction
    message = "I'm John Doe, and I'd like to return the shoes I bought last week."
    result = chat(client, message, EXTRACT_PROMPT)
    expected_facts = [
        "Customer name: John Doe",
        "Wants to return shoes",
        "Purchase made last week"
    ]
    assert semantic_match_multi(expected_facts, result["facts"]), f"Failed on basic facts: {result['facts']}"
    
    # Test case 2: Belief extraction
    message = "I think that we should build more housing."
    result = chat(client, message, EXTRACT_PROMPT)
    expected_belief = ["User believes that we should build more housing"]
    assert semantic_match_multi(expected_belief, result["facts"]), f"Failed on belief: {result['facts']}"
    
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
    assert semantic_match_multi(expected_facts, result["facts"]), f"Failed on multiple facts: {result['facts']}"


if __name__ == "__main__":
    # exercise the extraction prompt against GPT-4o
    test_extract()
