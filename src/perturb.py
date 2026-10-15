"""Prompt-template perturbation operators for the Prompt Fragility Index study.

Edits are applied ONLY to the instruction part of a prompt template (never to the
task input such as the review, article, question or passage).

Design for compounding analysis: every word-level operator draws its randomness
from an RNG seeded by (seed, edit_type, sentence_index), and sentence reordering
is applied after word-level edits. Therefore the prompt produced by a combination
of edits {e1, e2, ...} at seed s contains *exactly* the same local edits as the
individual variants e1@s, e2@s, ... — so the combined effect can be compared to
the sum of the individual effects of those very same edits.
"""
import hashlib
import random
import re

EDIT_TYPES = ["whitespace", "capitalization", "punctuation", "typo", "synonym", "reorder"]

# Canonical order in which word-level operators are composed.
WORD_LEVEL_ORDER = ["synonym", "typo", "punctuation", "capitalization"]

# Hand-curated synonym lexicon covering the vocabulary of the instruction templates.
# Replacements keep the meaning (near-synonyms), as a human prompt editor might.
SYNONYMS = {
    "expert": ["specialist", "skilled"], "assistant": ["helper", "aide"],
    "read": ["examine", "look at", "go through"], "carefully": ["attentively", "closely"],
    "provided": ["given", "supplied"], "below": ["underneath", "that follows"],
    "determine": ["decide", "identify", "figure out"], "whether": ["if"],
    "overall": ["general", "global"], "sentiment": ["feeling", "attitude", "opinion"],
    "expressed": ["conveyed", "communicated"], "answer": ["reply", "response"],
    "exactly": ["precisely", "strictly"], "one": ["a single"], "word": ["term"],
    "article": ["story", "piece"], "topic": ["subject", "category"],
    "belongs": ["corresponds", "pertains"], "possible": ["available", "candidate"],
    "topics": ["subjects", "categories"], "respond": ["reply", "answer"],
    "only": ["just", "solely"], "name": ["label", "title"],
    "choose": ["select", "pick"],
    "single": ["one", "sole"], "best": ["most suitable", "most appropriate"],
    "options": ["choices", "alternatives"], "option": ["choice", "alternative"],
    "correct": ["right", "accurate"], "knowledgeable": ["well-informed", "learned"],
    "tutor": ["teacher", "instructor"], "question": ["query", "problem"],
    "passage": ["text", "paragraph"], "short": ["brief", "concise"],
    "span": ["segment", "snippet"], "copied": ["taken", "extracted"],
    "directly": ["verbatim", "straight"], "without": ["with no", "omitting"],
    "explanation": ["justification", "elaboration"], "precise": ["accurate", "exact"],
    "comprehension": ["understanding"], "classifier": ["categorizer", "classification system"],
    "decide": ["determine", "judge"], "movie": ["film"], "review": ["critique"], "then": ["afterwards", "next"], "about": ["regarding", "concerning"],
   
    "analysis": ["assessment", "evaluation"],
    "multiple-choice": ["multiple-option"], "from": ["among"],
}

KEYBOARD_NEIGHBORS = {
    "a": "qs", "b": "vn", "c": "xv", "d": "sf", "e": "wr", "f": "dg", "g": "fh", "h": "gj",
    "i": "uo", "j": "hk", "k": "jl", "l": "k", "m": "n", "n": "bm", "o": "ip", "p": "o",
    "q": "w", "r": "et", "s": "ad", "t": "ry", "u": "yi", "v": "cb", "w": "qe", "x": "zc",
    "y": "tu", "z": "x",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-/']*")


def _rng(seed, edit, idx):
    """Deterministic RNG keyed by (seed, edit type, sentence index)."""
    h = hashlib.sha256(f"{seed}|{edit}|{idx}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _match_case(src, repl):
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


PHRASE_SYNONYMS = {"nothing else": ["nothing more", "no other text"]}


def synonym(sent, rng, protected, p=0.6):
    """Replace lexicon phrases/words with a near-synonym with probability p."""
    for ph, cands in PHRASE_SYNONYMS.items():
        if ph in sent and rng.random() < p:
            sent = sent.replace(ph, rng.choice(cands))
    def sub(m):
        w = m.group(0)
        lw = w.lower()
        if lw in protected or lw not in SYNONYMS:
            return w
        cands = [c for c in SYNONYMS[lw] if c != lw]
        if not cands or rng.random() >= p:
            return w
        return _match_case(w, rng.choice(cands))
    return WORD_RE.sub(sub, sent)


def typo(sent, rng, protected, p=0.15):
    """Introduce character-level typos (swap/drop/duplicate/neighbor key) in ~p of words (len>=4)."""
    def sub(m):
        w = m.group(0)
        if w.lower() in protected or len(w) < 4 or rng.random() >= p:
            return w
        i = rng.randrange(1, len(w) - 1)
        op = rng.choice(["swap", "drop", "dup", "key"])
        if op == "swap":
            return w[:i] + w[i + 1] + w[i] + w[i + 2:]
        if op == "drop":
            return w[:i] + w[i + 1:]
        if op == "dup":
            return w[:i] + w[i] + w[i:]
        nb = KEYBOARD_NEIGHBORS.get(w[i].lower())
        return w[:i] + (rng.choice(nb) if nb else w[i]) + w[i + 1:]
    return WORD_RE.sub(sub, sent)


def punctuation(sent, rng, protected, p_del=0.6, p_swap=0.2):
    """Delete or swap sentence punctuation (. , : ; ! ?). Slashes/hyphens/parentheses are kept."""
    out = []
    for ch in sent:
        if ch in ".,:;!?":
            r = rng.random()
            if r < p_del:
                continue
            if r < p_del + p_swap:
                out.append(rng.choice([c for c in ".,;:!" if c != ch]))
                continue
        out.append(ch)
    return "".join(out)


def capitalization(sent, rng, protected, seed=0):
    """Seed-dependent casing change: even seeds lowercase everything, odd seeds flip case of ~35% of words."""
    if seed % 2 == 0:
        return sent.lower()

    def sub(m):
        w = m.group(0)
        if rng.random() >= 0.35:
            return w
        return rng.choice([w.upper(), w.lower(), w.title(), w.swapcase()])
    return WORD_RE.sub(sub, sent)


def _whitespace_inner(sent, rng, p=0.3):
    """Inflate random inter-word spaces to 2-3 spaces or a tab."""
    return re.sub(" ", lambda m: rng.choice(["  ", "   ", " \t", "\t"]) if rng.random() < p else " ", sent)


WORD_OPS = {"synonym": synonym, "typo": typo, "punctuation": punctuation}


def build_instruction(sentences, edits, seed, protected, base_sep="\n\n"):
    """Apply a set of edits to the instruction sentences.

    Returns (instruction_text, separator_between_instruction_and_input).
    """
    edits = set(edits)
    protected = {w.lower() for w in protected}
    sents = list(sentences)
    for i, s in enumerate(sents):
        for e in WORD_LEVEL_ORDER:
            if e not in edits:
                continue
            rng = _rng(seed, e, i)
            if e == "capitalization":
                s = capitalization(s, rng, protected, seed=seed)
            else:
                s = WORD_OPS[e](s, rng, protected)
        if "whitespace" in edits:
            s = _whitespace_inner(s, _rng(seed, "whitespace", i))
        sents[i] = s
    order = list(range(len(sents)))
    if "reorder" in edits:
        rng = _rng(seed, "reorder", -1)
        while order == list(range(len(sents))):
            rng.shuffle(order)
    sents = [sents[j] for j in order]
    if "whitespace" in edits:
        rng = _rng(seed, "whitespace", -1)
        joiners = [rng.choice([" ", "  ", "\n", " \n", "   "]) for _ in range(len(sents) - 1)]
        text = sents[0] + "".join(j + s for j, s in zip(joiners, sents[1:]))
        text = rng.choice(["", " ", "\n"]) + text + rng.choice(["  ", "\t", " \n"])
        sep = rng.choice(["\n", "\n\n\n", "\n \n", "\n\n\n\n"])
    else:
        text = " ".join(sents)
        sep = base_sep
    return text, sep


def char_edit_distance_ratio(a, b):
    """Normalized Levenshtein distance between two strings (edit magnitude diagnostic)."""
    import difflib
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()
