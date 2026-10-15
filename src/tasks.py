"""Task definitions: base instruction templates, item sampling, and scoring.

Task types:
  classification : SST-2 (binary sentiment), AG News (4-way topic)
  multiple_choice: ARC-Challenge (4 options, letter answer)
  generation     : SQuAD v1.1 (free-form short-answer generation, token-F1 / EM)
"""
import json
import random
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets"


def _load_jsonl(name):
    return [json.loads(l) for l in open(DATA / name)]


# ----------------------------------------------------------------------------- scoring
def _norm_squad(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def squad_f1(pred, gold):
    p, g = _norm_squad(pred).split(), _norm_squad(gold).split()
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec)


def score_squad(output, golds):
    out = output.strip().split("\n")[0].strip().strip('"').rstrip(".")
    f1 = max(squad_f1(out, g) for g in golds)
    em = max(float(_norm_squad(out) == _norm_squad(g)) for g in golds)
    return {"score": f1, "em": em, "parsed": out}


def score_label(output, gold, label_patterns):
    """Return 1 if the earliest-mentioned label in the output equals gold."""
    low = output.lower()
    best, pos = None, 10 ** 9
    for lab, pats in label_patterns.items():
        for pat in pats:
            m = re.search(pat, low)
            if m and m.start() < pos:
                best, pos = lab, m.start()
    return {"score": float(best == gold), "em": float(best == gold), "parsed": best}


def score_letter(output, gold):
    """Leading option letter (any case), else first standalone UPPERCASE A-D (avoids the article 'a')."""
    out = output.strip().lstrip("*(")
    m = re.match(r"([A-Da-d])(?:[\)\.:,*]|\s|$)", out)
    if not m:
        m = re.search(r"(?:^|[^A-Za-z])\(?([A-D])(?:[\)\.:,*]|\s|$)", out)
    pred = m.group(1).upper() if m else None
    return {"score": float(pred == gold), "em": float(pred == gold), "parsed": pred}


# ----------------------------------------------------------------------------- tasks
SST2_LABELS = {"positive": [r"\bpositive\b"], "negative": [r"\bnegative\b"]}
AG_LABELS = {
    "World": [r"\bworld\b"], "Sports": [r"\bsports?\b"], "Business": [r"\bbusiness\b"],
    "Sci/Tech": [r"sci\s*/\s*tech", r"\bscience\b", r"\btechnology\b", r"\btech\b"],
}
AG_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

TASKS = {
    "sst2": {
        "type": "classification",
        "sentences": [
            "You are an expert sentiment analysis assistant.",
            "Read the movie review provided below carefully.",
            "Determine whether the overall sentiment expressed in the review is positive or negative.",
            "Answer with exactly one word: positive or negative.",
        ],
        "protected": ["positive", "negative"],
        "max_tokens": 4,
    },
    "agnews": {
        "type": "classification",
        "sentences": [
            "You are an expert news topic classifier.",
            "Read the news article provided below carefully.",
            "Decide which topic the article belongs to; the possible topics are World, Sports, Business, and Sci/Tech.",
            "Respond with only the topic name and nothing else.",
        ],
        "protected": ["world", "sports", "business", "sci/tech"],
        "max_tokens": 5,
    },
    "arc": {
        "type": "multiple_choice",
        "sentences": [
            "You are a knowledgeable science tutor.",
            "Read the multiple-choice question provided below carefully.",
            "Choose the single best answer from the options.",
            "Respond with only the letter of the correct option (A, B, C, or D).",
        ],
        "protected": ["a", "b", "c", "d"],
        "max_tokens": 8,
    },
    "squad": {
        "type": "generation",
        "sentences": [
            "You are a precise reading comprehension assistant.",
            "Read the passage provided below carefully and then answer the question about it.",
            "The answer is a short span copied directly from the passage.",
            "Respond with only the answer span, without any explanation.",
        ],
        "protected": [],
        "max_tokens": 24,
    },
}


def load_items(task, n, seed=42):
    """Sample n evaluation items (fixed seed). Returns list of dicts with id, input, gold."""
    rng = random.Random(seed)
    items = []
    if task == "sst2":
        rows = _load_jsonl("sst2_validation.jsonl")
        rng.shuffle(rows)
        for r in rows[:n]:
            items.append({"id": f"sst2-{r['idx']}", "input": f"Review: {r['sentence'].strip()}",
                          "gold": "positive" if r["label"] == 1 else "negative"})
    elif task == "agnews":
        rows = _load_jsonl("ag_news_test.jsonl")
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        for i in idx[:n]:
            r = rows[i]
            items.append({"id": f"ag-{i}", "input": f"Article: {r['text'].strip()}",
                          "gold": AG_NAMES[r["label"]]})
    elif task == "arc":
        rows = [r for r in _load_jsonl("arc_challenge_test.jsonl") if r["choices"]["label"] == ["A", "B", "C", "D"]]
        rng.shuffle(rows)
        for r in rows[:n]:
            opts = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
            items.append({"id": r["id"], "input": f"Question: {r['question']}\n{opts}", "gold": r["answerKey"]})
    elif task == "squad":
        # Short passages only (<= 110 words) to keep CPU prefill cost manageable.
        rows = [r for r in _load_jsonl("squad_validation.jsonl") if len(r["context"].split()) <= 110]
        seen, uniq = set(), []
        for r in rows:  # at most one question per passage for item independence
            if r["context"] not in seen:
                seen.add(r["context"])
                uniq.append(r)
        rng.shuffle(uniq)
        for r in uniq[:n]:
            items.append({"id": r["id"], "input": f"Passage: {r['context']}\n\nQuestion: {r['question']}",
                          "gold": r["answers"]["text"]})
    else:
        raise ValueError(task)
    assert len(items) == n, (task, len(items))
    return items


def score(task, output, gold):
    if task == "sst2":
        return score_label(output, gold, SST2_LABELS)
    if task == "agnews":
        return score_label(output, gold, AG_LABELS)
    if task == "arc":
        return score_letter(output, gold)
    if task == "squad":
        return score_squad(output, gold)
    raise ValueError(task)
