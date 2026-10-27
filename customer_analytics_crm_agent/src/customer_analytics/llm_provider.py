"""Optional LLM narration adapter. Deterministic analytics never depends on it."""
from __future__ import annotations
import json
import os
from typing import Any


def is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def narrate_computed_results(interpretations: dict[str, Any]) -> dict[str, Any]:
    """Ask an optional LLM to clarify provided aggregate interpretations only.

    Raw customer rows, identifiers, feedback texts and tables are deliberately not sent.
    On any import/API/parse failure the caller should retain deterministic interpretations.
    """
    if not is_configured():
        raise RuntimeError("OPENAI_API_KEY is not configured; no AI request was sent.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Optional dependency missing. Install openai to enable LLM narration.") from exc

    safe_payload = []
    for name, item in interpretations.items():
        safe_payload.append({
            "analysis": str(name),
            "overview": str(item.get("overview", ""))[:600],
            "key_findings": [str(x)[:500] for x in item.get("key_findings", [])[:8]],
            "business_implications": str(item.get("business_implications", ""))[:900],
            "recommendations": [str(x)[:500] for x in item.get("recommendations", [])[:8]],
            "kpis": [str(x)[:300] for x in item.get("kpis", [])[:8]],
            "limitations": [str(x)[:500] for x in item.get("limitations", [])[:8]],
        })
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role":"system","content":(
                "You are a careful business analyst. Clarify only the supplied aggregated findings and recommendations. "
                "Do not create or change metrics, numbers, p-values, dates, evidence sources or effect estimates. "
                "Do not assert causation. Preserve every limitation. Return a JSON object mapping each analysis name to "
                "an array named 'plain_language_explanation' and an array named 'implementation_checklist'.")},
            {"role":"user","content":json.dumps(safe_payload, ensure_ascii=False)}
        ],
        response_format={"type":"json_object"},
    )
    text = response.choices[0].message.content or "{}"
    return json.loads(text)
