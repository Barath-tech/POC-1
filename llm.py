import os
import openai
from dotenv import load_dotenv
load_dotenv()

# OpenRouter setup
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SUMMARIZER_MODEL = "gpt-oss-20b"

def gpt_summarize(text, max_tokens=50, model=SUMMARIZER_MODEL):
    print("🔹 Calling GPT for summarization...")
    prompt = f"Summarize the following incident in one short sentence:\n\n{text}"
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def gpt_category_name(text, max_tokens=10, model=SUMMARIZER_MODEL):
    print("🔹 Calling GPT for category naming...")
    prompt = f"""
You are an assistant that assigns short category names.

Task:
- Read the given incident description.
- Return ONLY a short category name (1–3 words).
- Do NOT include explanations, punctuation, or numbering.

Examples:
Incident: "Server down in Bangalore data center"
Category: Server Issue

Incident: "Email not syncing on Outlook"
Category: Email Issue

Incident: "Laptop battery not charging"
Category: Hardware Issue

Now categorize this incident:
{text}
Category:
"""
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()
