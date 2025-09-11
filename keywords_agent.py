# keywords_ai.py
import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_keywords(seed: str, language: str = "en", country: str = "US", n: int = 30) -> dict:
    prompt = f"""
You are an SEO keyword research assistant.
Generate {n} realistic search queries people in {country} ({language}) might type around the topic "{seed}".
For each query, label it as Informational, Transactional, or Navigational intent.
Return JSON with fields: keywords[], clusters{{informational[], transactional[], navigational[]}}, suggestions[].
- suggestions[] should propose 1–2 content pages with grouped keywords.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    try:
        data = json.loads(response.choices[0].message.content)
    except Exception:
        data = {"keywords": [], "clusters": {}, "suggestions": []}

    return {
        "seed": seed,
        "language": language,
        "country": country,
        **data
    }
