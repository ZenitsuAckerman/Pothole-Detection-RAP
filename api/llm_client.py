import os, logging
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("pothole-api")
client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Bounded retries: max 3 attempts, short exponential backoff (1s, 2s, 4s cap).
# Deliberately NOT unbounded — a live verbal-defense demo can't afford a request
# that hangs for 30+ seconds waiting on Groq. Total worst case ~7s before failing safe.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _call_groq(prompt: str, max_tokens: int, json_mode: bool):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        timeout=5,
        **kwargs,
    )
    return resp.choices[0].message.content

def call_llm(prompt: str, max_tokens: int = 300, json_mode: bool = False) -> str | None:
    """Returns None if all retries are exhausted — callers must fail safe
    (never crash the endpoint on a None)."""
    try:
        return _call_groq(prompt, max_tokens, json_mode)
    except Exception as e:
        logger.error(f"Groq call failed after retries: {e}")
        return None

