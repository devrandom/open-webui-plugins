import os

EXTRACT_PROMPT = f"""
Please extract any facts stated by the user and provide the information in a JSON format.
Include beliefs by recording them as such.

Input: Hi.
Output: {{"facts" : []}}

Input: The weather is nice today.
Output: {{"facts" : []}}

Input: My order #12345 hasn't arrived yet.
Output: {{"facts" : ["Order #12345 not received"]}}

Input: I'm John Doe, and I'd like to return the shoes I bought last week.
Output: {{"facts" : ["Customer name: John Doe", "Wants to return shoes", "Purchase made last week"]}}

Input: I think we should keep our desks tidy.
Output: {{"facts" : ["User believes that we should keep our desks tidy"]}}

Return the facts in a json format as shown above.
"""


def test_extract():
    client = make_client()
    message = "I'm John Doe, and I'd like to return the shoes I bought last week."
    result = chat(client, message)
    expected_facts = [
        "Customer name: John Doe",
        "Wants to return shoes",
        "Purchase made last week"
    ]
    assert semantic_match(". ".join(result["facts"]), ". ".join(expected_facts)), result["facts"]
    message = "I think that we should build more housing."
    result = chat(client, message)
    assert semantic_match(".".join(result["facts"]), "User believes that we should build more housing."), result["facts"]


def semantic_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """
    Uses SentenceTransformer embeddings and cosine similarity
    to determine if two texts are semantically similar.
    """
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([text1, text2])
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_score >= threshold

def chat(client, message):
    import json

    messages = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": message}
    ]
    params = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 100,
    }
    response = client.chat.completions.create(**params)
    result = json.loads(response.choices[0].message.content)
    return result


def make_client():
    import openai
    from dotenv import load_dotenv
    load_dotenv(verbose=True)
    api_key = os.environ.get("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    return client


if __name__ == "__main__":
    # exercise the extraction prompt against GPT-4o
    test_extract()
