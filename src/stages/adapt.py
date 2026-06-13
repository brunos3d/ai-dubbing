"""Stage 6.5 - Speech adaptation for naturalness and tone."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.logging import get_logger, stage_banner
from ..utils.vram import free_vram, log_vram

LOG = get_logger("ai-dubbing.adapt")


class AdaptStage:
    """Refines translated text to sound more natural and spoken-language oriented."""

    name = "adapt"

    def __init__(
        self,
        workdir: Path,
        model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        device: str = "cuda",
        mode: str = "YouTube Narrator",
    ):
        self.workdir = Path(workdir)
        self.model_id = model_id
        self.device = device
        self.mode = mode
        self.model = None
        self.tokenizer = None

    def outputs(self) -> List[Path]:
        return [self.workdir / "adapted_transcript.json"]

    def _load_model(self):
        if self.model is None:
            LOG.info(f"Loading adaptation model: {self.model_id} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
            )

    def _adapt_text(self, text: str, original_text: str) -> str:
        self._load_model()
        
        prompt = (
            f"You are a professional {self.mode}. "
            f"Rewrite the following translated text to sound more natural and spoken-language oriented. "
            f"Preserve the meaning and pacing. DO NOT translate entities that are in ALL CAPS or placeholders like __ENT_0__.\n"
            f"Original (Source): {original_text}\n"
            f"Translated: {text}\n"
            f"Adapted:"
        )
        
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response.strip().split("\n")[0] # Take first line

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        stage_banner(LOG, 6, 12, "Speech Adaptation")
        translated_path = Path(context["translated_path"])
        out_path = self.workdir / "adapted_transcript.json"
        
        segments = json.loads(translated_path.read_text(encoding="utf-8"))
        adapted_segments = []
        
        total = len(segments)
        for i, seg in enumerate(segments):
            text = (seg.get("text") or "").strip()
            orig = (seg.get("source_text") or "").strip()
            
            if not text or seg.get("is_non_speech"):
                adapted_segments.append(dict(seg))
                continue
                
            try:
                adapted = self._adapt_text(text, orig)
                # Fallback if model hallucinated empty or garbage
                if not adapted or len(adapted) < 3:
                    adapted = text
            except Exception as exc:
                LOG.warning(f"Adaptation failed for segment {i}: {exc}")
                adapted = text
                
            new = dict(seg)
            new["translated_text"] = text # Keep intermediate
            new["text"] = adapted
            
            # Diagnostics
            src_words = len(orig.split())
            tgt_words = len(adapted.split())
            LOG.debug(f"Seg {i+1}: {text[:30]}... -> {adapted[:30]}... ({tgt_words-src_words} word delta)")
            
            adapted_segments.append(new)
            
            if (i + 1) % 10 == 0 or i == total - 1:
                LOG.info(f"Adapted {i + 1}/{total} segments")
                
        out_path.write_text(json.dumps(adapted_segments, indent=2, ensure_ascii=False))
        LOG.info(f"Adapted transcript -> {out_path}")
        
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        free_vram()
        log_vram(LOG)
        
        return {"adapted_path": str(out_path), "num_segments": len(adapted_segments)}
