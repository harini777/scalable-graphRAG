import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional
import requests

from src.config import settings
from src.models import RetrievalResult, Citation, AnswerResponse

logger = logging.getLogger(__name__)

CACHE_PATH = settings.CACHE_DIR / "llm_cache.json"

class LLMCache:
    def __init__(self, cache_file=CACHE_PATH):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")

    def get(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(prompt_hash)

    def set(self, prompt_hash: str, response: Dict[str, Any]):
        self.cache[prompt_hash] = response
        self._save()

llm_cache = LLMCache()

class AnswerGenerator:
    def __init__(
        self,
        provider: str = settings.LLM_PROVIDER,
        model: str = settings.LLM_MODEL
    ):
        self.provider = provider.lower()
        if self.provider == "groq" and model in ["gpt-4o-mini", "gpt-4o", "llama-3.3-70b-versatile"]:
            self.model = "openai/gpt-oss-120b"
        else:
            self.model = model

    def _hash_request(self, system_prompt: str, user_prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.provider.encode("utf-8"))
        h.update(self.model.encode("utf-8"))
        h.update(system_prompt.encode("utf-8"))
        h.update(user_prompt.encode("utf-8"))
        return h.hexdigest()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        req_hash = self._hash_request(system_prompt, user_prompt)
        cached = llm_cache.get(req_hash)
        if cached:
            return cached

        # Fallback Mock / Offline Mode
        if self.provider == "mock" or (not settings.OPENAI_API_KEY and not settings.GEMINI_API_KEY and not settings.ANTHROPIC_API_KEY and not settings.GROQ_API_KEY):
            res = self._mock_response(user_prompt)
            llm_cache.set(req_hash, res)
            return res

        # OpenAI compatible (OpenAI, Groq)
        if self.provider in ["openai", "groq"]:
            api_key = settings.OPENAI_API_KEY if self.provider == "openai" else settings.GROQ_API_KEY
            base_url = "https://api.openai.com/v1" if self.provider == "openai" else "https://api.groq.com/openai/v1"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0
            }
            try:
                resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                answer_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                result = {
                    "text": answer_text,
                    "tokens": usage.get("total_tokens", 0)
                }
                llm_cache.set(req_hash, result)
                return result
            except Exception as e:
                err_msg = resp.text if 'resp' in locals() and hasattr(resp, 'text') else str(e)
                logger.error(f"Error calling {self.provider} API: {err_msg}")
                raise RuntimeError(f"Error from {self.provider} API: {err_msg}") from e

        # Ollama local endpoint
        elif self.provider == "ollama":
            url = f"{settings.OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            result = {
                "text": data.get("response", ""),
                "tokens": data.get("eval_count", 0)
            }
            llm_cache.set(req_hash, result)
            return result

        # Default fallback
        res = self._mock_response(user_prompt)
        llm_cache.set(req_hash, res)
        return res

    def _mock_response(self, user_prompt: str) -> Dict[str, Any]:
        """Deterministic fallback when no external API key is active."""
        # Simple extraction of context info to produce a faithful mock response with citations
        lines = [l for l in user_prompt.split("\n") if "Paper [" in l or "Context [" in l]
        citations = []
        for line in lines:
            if "[" in line and "]" in line:
                start = line.find("[") + 1
                end = line.find("]")
                citations.append(line[start:end])

        citation_str = f" [{citations[0]}]" if citations else ""
        return {
            "text": f"Based on the provided research context{citation_str}, the relevant methods and evaluation results indicate that the adaptation approaches achieve targeted performance gains with significantly reduced parameter overhead.",
            "tokens": 45
        }

    def generate_answer(
        self,
        query: str,
        retrieved_results: List[RetrievalResult],
        system_name: str = "plain_rag",
        subgraph: Optional[Dict[str, Any]] = None
    ) -> AnswerResponse:
        start_time = time.time()

        # Build prompt context with strict citation format
        context_blocks = []
        for res in retrieved_results:
            paper_id = res.paper_id or res.metadata.get("arxiv_id", "Unknown")
            title = res.metadata.get("title", "")
            context_blocks.append(f"--- Document [{paper_id}] ({title}) ---\n{res.text}")

        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert scientific researcher. Answer the question using ONLY the provided research context.\n"
            "Rules:\n"
            "1. Every factual statement must cite its source in brackets, e.g. [2106.09685].\n"
            "2. If the answer cannot be determined from the evidence, state clearly what is known and what is missing.\n"
            "3. Be concise, direct, and technically precise."
        )

        user_prompt = f"Question: {query}\n\nEvidence Context:\n{context_str}\n\nAnswer:"

        llm_out = self._call_llm(system_prompt, user_prompt)
        elapsed_ms = (time.time() - start_time) * 1000

        # Extract citations
        answer_text = llm_out["text"]
        citations: List[Citation] = []
        for res in retrieved_results:
            pid = res.paper_id or res.metadata.get("arxiv_id", "")
            if pid and f"[{pid}]" in answer_text:
                citations.append(
                    Citation(
                        paper_id=pid,
                        paper_title=res.metadata.get("title", ""),
                        supporting_text=res.text[:200]
                    )
                )

        return AnswerResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            retrieved_results=retrieved_results,
            subgraph=subgraph,
            system_name=system_name,
            latency_ms=elapsed_ms,
            total_tokens=llm_out.get("tokens", 0)
        )
