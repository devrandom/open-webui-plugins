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
Output: {{"facts": ["User's name is John Doe", "User wants to return shoes", "The purchase was made last week", "Reason for return: shoes too small"]}}

Input: I think we should keep our desks tidy to improve productivity.
Output: {{"facts": ["User believes that we should keep our desks tidy", "User believes tidy desks improve productivity"]}}

Input: I'd prefer to be contacted by email at john@example.com rather than by phone.
Output: {{"facts": ["User's mail address: john@example.com", "User prefers email contact over phone contact"]}}

Always respond with valid JSON containing only the "facts" array.
"""

