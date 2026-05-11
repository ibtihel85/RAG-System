from __future__ import annotations

import logging
from typing import List, Optional

import torch

from src.data.schema import Document

logger = logging.getLogger(__name__)

_FALLBACK_ANSWER = "I don't know."
_MAX_PROMPT_TOKENS = 2048
_MAX_CHARS_PER_DOC = 500


def _trunc_at_sentence(text: str, max_chars: int) -> str:
    """Truncate *text* at the last sentence boundary within *max_chars*."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in [". ", "! ", "? "]:
        pos = truncated.rfind(sep)
        if pos > int(max_chars * 0.5):
            return truncated[: pos + 1]
    return truncated


class AnswerGenerator:

    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)

        logger.info("Loading generator tokeniser: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        logger.info("Loading generator model (4-bit): %s", model_name)
        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=hf_token,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                token=hf_token,
            )

        self.model.eval()
        logger.info("Generator model loaded.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a factual QA assistant. Answer with the shortest "
                        "possible correct span. 1-8 words maximum. No preamble."
                    ),
                },
                {"role": "user", "content": f"Question: {query}\nAnswer:"},
            ]
        else:
            context_parts = []
            for i, doc in enumerate(context_docs, start=1):
                passage = _trunc_at_sentence(doc.body, _MAX_CHARS_PER_DOC)
                context_parts.append(f"[P{i}] {passage}")
            context_block = "\n\n".join(context_parts)

            system_msg = (
                "You are a precise extractive QA assistant. "
                "Find the answer span in the passages and output it VERBATIM.\n\n"
                "RULES:\n"
                "  1. Output ONLY the answer span — no explanation, no preamble.\n"
                "  2. Copy exact words from a passage. Do NOT paraphrase.\n"
                "  3. Answer length: 1-8 words (names, dates, numbers, short phrases).\n"
                "  4. Do NOT start with: The answer is, Based on, According to.\n"
                "  5. Do NOT output a full sentence — only the answer span itself.\n"
                "  6. If absent from all passages: output exactly: I don't know\n\n"
                "EXAMPLES:\n"
                "  Q: What year did World War II end?    -> 1945\n"
                "  Q: Who wrote Hamlet?                 -> William Shakespeare\n"
                "  Q: What is the capital of France?    -> Paris\n"
                "  Q: How many bones in the human body? -> 206\n"
            )
            user_msg = (
                f"Passages:\n{context_block}\n\n"
                f"Question: {query}\n\n"
                "Answer (verbatim span from passages, 1-8 words):"
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        query: str,
        context_docs: List[Document],
        max_new_tokens: int = 30,
        temperature: float = 0.0,
    ) -> str:
        prompt = self._build_prompt(query, context_docs)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_PROMPT_TOKENS,
            padding=False,
        ).to(next(self.model.parameters()).device)

        input_len = inputs["input_ids"].shape[1]

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = output_ids[0][input_len:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return self._clean(raw)

    @staticmethod
    def _clean(text: str) -> str:
        """Strip common preamble patterns the model might still produce."""
        prefixes = [
            "the answer is",
            "based on",
            "according to",
            "answer:",
            "answer :",
        ]
        lower = text.lower()
        for pref in prefixes:
            if lower.startswith(pref):
                text = text[len(pref):].strip(" :.,-")
                break

        # Truncate to first newline (model sometimes adds explanation)
        text = text.split("\n")[0].strip()

        return text if text else _FALLBACK_ANSWER
