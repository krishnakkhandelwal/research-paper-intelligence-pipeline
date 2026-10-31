"""
Enrichment stage: fills the LLM-derived fields of PaperSchema
(problem_statement, novel_contribution, datasets, models, metrics,
limitations, future_work).

Primary path: OpenAI chat completion in JSON mode, temperature 0, with the
output validated and coerced to the schema's types (one retry on bad JSON).
Fallback path: a rule-based extractor, used when there is no API key, the
provider is not 'openai', or the LLM call keeps failing. This keeps the
pipeline runnable offline and for free.
"""

import json
import re
from typing import Dict, List

from config.settings import settings
from src.utils.logger import get_logger, log_stage
from src.utils.schema import PaperSchema

logger = get_logger("llm_extractor")

LIST_FIELDS = ("datasets", "models", "metrics")
TEXT_FIELDS = ("problem_statement", "novel_contribution", "limitations", "future_work")

# Sections most likely to contain the information we want, in priority order.
PRIORITY_KEYWORDS = ["abstract", "introduction", "conclusion", "limitation", "future",
                     "discussion", "experiment", "result", "evaluation", "dataset", "method"]

SYSTEM_PROMPT = (
    "You extract structured information from a research paper. Use ONLY the text provided. "
    "If something is not stated, use an empty string or empty list; never guess. "
    "Return a single JSON object with exactly these keys: "
    "problem_statement (string, 1-2 sentences), novel_contribution (string, 1-2 sentences), "
    "datasets (list of dataset names), models (list of model/method names), "
    "metrics (list of evaluation metrics), limitations (string), future_work (string)."
)


# ---------- context building ----------

def build_context(paper: PaperSchema, max_chars: int) -> str:
    """Packs the most informative sections first, until max_chars is reached."""
    parts: List[str] = []
    used = set()
    if paper.title:
        parts.append(f"TITLE: {paper.title}")

    def take(name: str, text: str):
        remaining = max_chars - sum(len(p) for p in parts)
        if remaining > 200 and name not in used:
            parts.append(f"[{name.upper()}]\n{text[:min(len(text), remaining, 3500)]}")
            used.add(name)

    if paper.abstract:
        take("abstract", paper.abstract)
    for kw in PRIORITY_KEYWORDS:
        for name, text in paper.sections.items():
            if kw in name.lower() and name.lower() not in ("references", "bibliography"):
                take(name, text)
    if len(parts) <= 1 and paper.raw_text:  # no sections (e.g. scanned PDF)
        take("full text", paper.raw_text)
    return "\n\n".join(parts)


# ---------- LLM path ----------

def _coerce(data: Dict) -> Dict:
    out: Dict = {}
    for f in TEXT_FIELDS:
        v = data.get(f, "")
        out[f] = v.strip() if isinstance(v, str) else ""
    for f in LIST_FIELDS:
        v = data.get(f, [])
        items = [str(x).strip() for x in v] if isinstance(v, list) else []
        out[f] = list(dict.fromkeys(i for i in items if i))  # de-duplicate, keep order
    return out


def _llm_extract(context: str) -> Dict:
    from openai import OpenAI  # imported lazily so the fallback works without the package configured

    client = OpenAI(api_key=settings.openai_api_key)
    last_error = None
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": context}],
            )
            usage = resp.usage
            logger.info(f"LLM tokens: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
            return _coerce(json.loads(resp.choices[0].message.content))
        except Exception as e:  # network, rate limit, invalid JSON
            last_error = e
            logger.warning(f"LLM extraction attempt {attempt} failed: {e}")
    raise RuntimeError(f"LLM extraction failed: {last_error}")


# ---------- heuristic fallback ----------

KNOWN_DATASETS = ["ImageNet", "COCO", "MNIST", "CIFAR-10", "CIFAR-100", "SQuAD", "GLUE", "SuperGLUE",
                  "WMT", "Penn Treebank", "WikiText", "Common Crawl", "OpenWebText", "LibriSpeech",
                  "MS MARCO", "Wikipedia", "BookCorpus", "Pascal VOC", "Cityscapes", "KITTI"]
KNOWN_MODELS = ["Transformer", "BERT", "GPT", "RoBERTa", "T5", "LSTM", "GRU", "ResNet", "VGG",
                "CNN", "RNN", "U-Net", "YOLO", "ViT", "CLIP", "LLaMA", "Random Forest",
                "XGBoost", "SVM", "GAN", "VAE", "Diffusion"]
KNOWN_METRICS = ["accuracy", "F1", "precision", "recall", "BLEU", "ROUGE", "AUC", "mAP",
                 "perplexity", "RMSE", "MAE", "IoU", "WER", "exact match"]


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if s.strip()]


def _find_terms(text: str, vocab: List[str]) -> List[str]:
    return [t for t in vocab if re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", text, re.I if t.islower() else 0)]


def _first_match(sentences: List[str], pattern: str) -> str:
    rx = re.compile(pattern, re.I)
    return next((s for s in sentences if rx.search(s)), "")


def heuristic_extract(paper: PaperSchema) -> Dict:
    abstract_sents = _sentences(paper.abstract) or _sentences(paper.raw_text[:3000])
    all_text = " ".join([paper.abstract] + list(paper.sections.values())) or paper.raw_text
    all_sents = _sentences(all_text)

    limitations = next((v for k, v in paper.sections.items() if "limitation" in k.lower()), "")
    future = next((v for k, v in paper.sections.items() if "future" in k.lower()), "")

    return _coerce({
        "problem_statement": " ".join(abstract_sents[:2]),
        "novel_contribution": _first_match(
            abstract_sents + all_sents,
            r"\bwe (propose|introduce|present|develop|show)\b|this (paper|work) (proposes|introduces|presents)|our (main )?contribution"),
        "datasets": _find_terms(all_text, KNOWN_DATASETS),
        "models": _find_terms(all_text, KNOWN_MODELS),
        "metrics": _find_terms(all_text, KNOWN_METRICS),
        "limitations": limitations[:600] or _first_match(all_sents, r"\blimitation"),
        "future_work": future[:600] or _first_match(all_sents, r"future work|in future|future research"),
    })


# ---------- public stage ----------

@log_stage("enrichment")
def enrich_paper(paper: PaperSchema, paper_id: str = "") -> PaperSchema:
    """Mutates and returns the paper with the LLM-derived fields filled in."""
    use_llm = settings.llm_provider == "openai" and bool(settings.openai_api_key)
    result = None

    if use_llm:
        try:
            result = _llm_extract(build_context(paper, settings.llm_max_context_chars))
        except RuntimeError as e:
            logger.error(f"{e}; falling back to heuristic extractor")
    else:
        logger.warning("No usable LLM configured (provider/key); using heuristic extractor")

    if result is None:
        result = heuristic_extract(paper)

    for f in TEXT_FIELDS + LIST_FIELDS:
        setattr(paper, f, result[f])
    return paper
