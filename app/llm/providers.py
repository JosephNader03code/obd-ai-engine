"""Pluggable LLM providers. Default = `mock` (offline, deterministic).

Environment variables::

    LLM_PROVIDER = mock | anthropic | openai | ollama
    LLM_MODEL    = model id (defaults per provider)
    LLM_BASE_URL = for openai-compatible servers (OpenAI, Ollama, LM Studio, Groq, OpenRouter...)
    LLM_API_KEY  = key (or ANTHROPIC_API_KEY / OPENAI_API_KEY)
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import httpx

SYSTEM_PROMPT_AR = """أنت مساعد تشخيص سيارات داخل نظام OBD-II. دورك محدود وواضح:
- أنت لا تخترع أعطالاً ولا تتخذ قراراً ميكانيكياً من نفسك.
- تفهم نتائج الـ Rule Engine (R*) ونتائج الـ ML (M*) والمعلومات المسترجعة (K*) والأكواد (D*) والقياسات (features).
- تنظم الأدلة، وتشرح الأسباب المحتملة بلغة عربية مصرية بسيطة يفهمها صاحب العربية والميكانيكي.
- تقترح خطوات الفحص التالية من قائمة الاختبارات المعطاة فقط (tests).

قواعد إلزامية:
1. كل سبب محتمل لازم يستشهد بـ id واحد على الأقل من نوع R أو M أو D. لا تذكر سبباً غير مدعوم.
2. لا تذكر أي كود عطل غير موجود في الأدلة.
3. لو الأدلة متعارضة (مثلاً R يقول تسريب هواء و P0420 موجود) وضّح الترتيب المنطقي للفحص.
4. لو البيانات ناقصة قل ذلك صراحة في limitations.
5. نتائج ML مدربة على بيانات محاكاة: اعتبرها دليلاً مساعداً وليس حاسماً.
6. أرجع JSON فقط بالشكل التالي بدون أي نص آخر:
{"summary": "...",
 "likely_causes": [{"cause": "...", "likelihood": "high|medium|low", "evidence": ["R1","K1"], "why": "..."}],
 "next_steps": [{"step": "...", "test_id": "T_... أو null", "evidence": ["R1"]}],
 "safety": "...",
 "limitations": "..."}"""

SYSTEM_PROMPT_EN = """You are a vehicle diagnostics assistant within an OBD-II system. Your role is strictly limited:
- You do not invent faults or make mechanical decisions on your own.
- You interpret Rule Engine results (R*), ML model results (M*), retrieved knowledge (K*), DTCs (D*) and measured features.
- You organize the evidence and explain likely causes in clear, simple English for both vehicle owners and mechanics.
- You suggest next diagnostic steps ONLY from the provided tests list.

Mandatory rules:
1. Every likely cause must cite at least one evidence ID of type R, M, or D. Do not mention unsupported causes.
2. Do not mention any DTC code not present in the evidence.
3. If evidence is contradictory, explain the logical order of diagnosis.
4. If data is missing, state it explicitly in limitations.
5. ML results are trained on simulation data: treat them as supporting evidence, not decisive.
6. Return JSON only in this exact format with no other text:
{"summary": "...",
 "likely_causes": [{"cause": "...", "likelihood": "high|medium|low", "evidence": ["R1","K1"], "why": "..."}],
 "next_steps": [{"step": "...", "test_id": "T_... or null", "evidence": ["R1"]}],
 "safety": "...",
 "limitations": "..."}"""

SYSTEM_PROMPT = SYSTEM_PROMPT_AR


class LLMProvider:
    name = "base"

    def complete(self, evidence_pkg: Dict) -> str:
        raise NotImplementedError

    @staticmethod
    def user_prompt(evidence_pkg: Dict) -> str:
        return "حزمة الأدلة (JSON):\n" + json.dumps(evidence_pkg, ensure_ascii=False, indent=1)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or "claude-sonnet-5-5"
        self.key = api_key or os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not self.key:
            raise RuntimeError("ANTHROPIC_API_KEY missing")

    def complete(self, evidence_pkg: Dict) -> str:
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=90, headers={
            "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.model, "max_tokens": 2000, "system": SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": self.user_prompt(evidence_pkg)}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base = (base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4o-mini"
        self.key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "none"

    def complete(self, evidence_pkg: Dict) -> str:
        r = httpx.post(f"{self.base}/chat/completions", timeout=180,
                       headers={"Authorization": f"Bearer {self.key}"},
                       json={"model": self.model, "temperature": 0.1,
                             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": self.user_prompt(evidence_pkg)}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def get_provider() -> Optional[LLMProvider]:
    """Returns None for the mock provider (handled by the explainer)."""
    kind = os.getenv("LLM_PROVIDER", "mock").lower()
    model = os.getenv("LLM_MODEL") or None
    if kind == "anthropic":
        return AnthropicProvider(model)
    if kind == "openai":
        return OpenAICompatProvider(model)
    if kind == "ollama":
        return OpenAICompatProvider(model or "qwen2.5:7b", os.getenv("LLM_BASE_URL") or "http://localhost:11434/v1")
    return None
