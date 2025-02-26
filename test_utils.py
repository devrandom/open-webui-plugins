import os
import json
from sentence_transformers import SentenceTransformer, util

def semantic_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Uses SentenceTransformer embeddings and cosine similarity
    to determine if two texts are semantically similar.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([text1, text2])
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_score >= threshold


def semantic_match_multi(expected_facts: list, actual_facts: list, threshold: float = 0.8, debug: bool = False) -> bool:
    """
    Establishes a 1:1 mapping between expected facts and actual facts based on semantic similarity.
    
    Args:
        expected_facts: List of expected fact strings
        actual_facts: List of actual fact strings
        threshold: Minimum similarity threshold (0-1)
        debug: If True, print detailed matching information
        
    Returns:
        bool: True if there is a 1:1 mapping where each expected fact matches an actual fact
    """
    import numpy as np
    
    # Check if we have enough actual facts
    if len(actual_facts) < len(expected_facts):
        if debug:
            print(f"Not enough actual facts: {len(actual_facts)} < {len(expected_facts)}")
        return False
    
    # Get model and encode all texts at once (more efficient)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    all_texts = expected_facts + actual_facts
    embeddings = model.encode(all_texts)
    
    # Split embeddings into expected and actual
    expected_embeddings = embeddings[:len(expected_facts)]
    actual_embeddings = embeddings[len(expected_facts):]
    
    # Calculate similarity matrix
    # Each row is an expected fact, each column is an actual fact
    similarity_matrix = util.cos_sim(expected_embeddings, actual_embeddings).cpu().numpy()
    
    # Use a greedy matching algorithm to find the best 1:1 mapping
    matched_actual_indices = set()
    matches = []  # Store matches for debugging
    unmatched = []  # Store unmatched facts
    
    for expected_idx in range(len(expected_facts)):
        best_match_idx = -1
        best_match_score = threshold  # Must be at least threshold
        
        for actual_idx in range(len(actual_facts)):
            if actual_idx in matched_actual_indices:
                continue  # Skip already matched facts
                
            score = similarity_matrix[expected_idx, actual_idx]
            if score > best_match_score:
                best_match_score = score
                best_match_idx = actual_idx
        
        if best_match_idx == -1:
            unmatched.append(expected_facts[expected_idx])
            if debug:
                print(f"No match found for: {expected_facts[expected_idx]}")
                print(f"Available facts: {[f for i, f in enumerate(actual_facts) if i not in matched_actual_indices]}")
            continue  # Skip to next expected fact
            
        matches.append((expected_facts[expected_idx], actual_facts[best_match_idx], best_match_score))
        matched_actual_indices.add(best_match_idx)
    
    if debug and matches:
        print("\nMatched facts:")
        for expected, actual, score in matches:
            print(f"Expected: '{expected}' -> Actual: '{actual}' (score: {score:.3f})")
    
    # Return True only if all expected facts were matched
    success = len(unmatched) == 0
    
    if debug and not success:
        print("\nUnmatched facts:")
        for fact in unmatched:
            print(f"- {fact}")
    
    return success


def chat(client, message, prompt, model="gpt-4o-mini", max_tokens=150, temperature=0.0):
    """
    Send a message to the LLM and extract facts from the response.
    
    Args:
        client: OpenAI client
        message: User message to extract facts from
        prompt: System prompt to use for the extraction
        model: LLM model to use
        max_tokens: Maximum tokens in the response
        temperature: Response temperature (lower for more consistent outputs)
    
    Returns:
        Dictionary containing extracted facts
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": message}
    ]
    
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        
        # Handle potential JSON parsing errors
        try:
            result = json.loads(content)
            # Ensure result has the expected format
            if "facts" not in result:
                result = {"facts": []}
            return result
        except json.JSONDecodeError:
            print(f"Error parsing JSON response: {content}")
            return {"facts": []}
            
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return {"facts": []}


def make_client():
    """
    Create and configure an OpenAI client instance.
    
    Returns:
        Configured OpenAI client
    """
    import openai
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv(verbose=True)
    
    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable")
    
    # Create client
    client = openai.OpenAI(api_key=api_key)
    
    return client

