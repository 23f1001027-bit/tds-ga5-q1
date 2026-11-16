"""
RAG Colosseum - retrieval + extractive QA service over a 50-passage knowledge base.

POST /query  {"question": str, "top_k": int}
  -> {"answer": str|null, "sources": [int], "scores": [float], "confidence": float}

Retrieval is a hybrid of word TF-IDF, character n-gram TF-IDF and a "topic sentence"
match (the first sentence of each passage names its subject), plus an attribute-aware
boost (e.g. a question about a port favours passages that state a port).
Answers are extracted verbatim from the knowledge base, so injected instructions in
the question can never leak into the answer.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import TfidfVectorizer

KB_PATH = Path(os.environ.get("KB_PATH", Path(__file__).with_name("knowledge_base.json")))
PASSAGES: list[str] = json.loads(KB_PATH.read_text(encoding="utf-8"))
KB_LOWER = "\n".join(PASSAGES).lower()

# --------------------------------------------------------------------------- text utils
STOP = set("""
a an the of to in on for by with and or is are was were be been being do does did what
which who whom whose how when where why that this these those it its as at from into
than then there their they them i you we he she me my our your us can could should would
will shall may might must per each any all some about over under between across after
before during if whether not no yes so such also only just very more most much many
tell explain describe give list name according kb knowledge base passage context
please answer question value number exact exactly specific specified state stated
""".split())

ALIASES = {
    "postgres": "postgresql", "pg": "postgresql", "mongo": "mongodb", "innodb": "innodb",
    "hlc": "hybrid logical clocks", "hlcs": "hybrid logical clocks", "lsm": "log-structured merge",
    "sgd": "stochastic gradient descent", "mlp": "multi-layer perceptron", "mlps": "multi-layer perceptrons",
    "svm": "support vector machines", "svms": "support vector machines", "resnet": "residual convolutional resnet",
    "2pc": "two-phase commit", "3pc": "three-phase commit", "wal": "write-ahead logging",
    "mdns": "multicast dns", "ospf": "open shortest path first", "bgp": "border gateway protocol",
    "tcp": "transmission control protocol tcp", "dns": "domain name system dns", "ecc": "elliptic curve",
    "aes": "advanced encryption standard aes", "jwt": "json web tokens jwt", "jwts": "json web tokens jwts",
    "pkce": "proof key for code exchange pkce", "waf": "web application firewalls", "hrw": "rendezvous hashing",
    "layernorm": "layer normalization", "batchnorm": "batch normalization",
    "rsa": "rsa", "tls": "transport layer security tls",
    "ttl": "time-to-live", "pfs": "perfect forward secrecy", "csp": "content security policy",
    "ece": "expected calibration error", "rbf": "radial basis function", "hrw": "highest random weight",
    "iv": "initialization vectors", "ivs": "initialization vectors", "ca": "certificate authorities",
    "cas": "certificate authorities", "dh": "diffie-hellman", "dhe": "ephemeral diffie-hellman",
    "mac": "message authentication codes", "macs": "message authentication codes", "lr": "learning rate",
    "ssd": "sequential disk", "mvcc": "multi-version concurrency control", "csrf": "cross-site request forgery",
    "xss": "cross-site scripting", "sqli": "sql injection", "qp": "quadratic programs", "l2": "l2",
    "nlp": "natural language processing", "cv": "computer vision", "gelu": "gelu", "rope": "rotary position",
    "acid": "acid", "sst": "sstable", "kv": "key-value", "rtt": "round trip", "hol": "head-of-line blocking",
    "ack": "acknowledge", "acks": "acknowledgment", "lb": "load", "ddos": "rate limiting",
}


def norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(\d+)\s*(?:milliseconds?|msec|ms)\b", r"\1ms", t)
    t = re.sub(r"(\d+)\s*-?\s*bits?\b", r"\1 bits", t)
    return " ".join(re.findall(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*", t))


SYNONYMS = [
    (r"\bstructur\w*|\bformat(?:ted)?\b|\bcompos\w*|\bmade (?:up )?of\b", "comprise segments"),
    (r"\bmachines?\b|\bservers?\b", "nodes"),
    (r"\bspeed(?:s)? up\b|\bfaster\b", "accelerate"),
    (r"\bhow (?:large|big)\b", "size bits"),
    (r"\bwait(?:s|ing)? for\b|\bwaits?\b", "wait timeout"),
    (r"\bconsensus\b", "consensus quorum"),
]


def expand_query(q: str) -> str:
    ql = q.lower()
    extra = [ALIASES[w] for w in re.findall(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*", ql) if w in ALIASES]
    for a in re.findall(r"\b[A-Z]{2,6}s?\b", q):
        k = a.lower().rstrip("s") if a.endswith("s") and len(a) > 2 else a.lower()
        if len(k) >= 4 and k not in ALIASES and k in ACRONYMS and not re.search(r"(?<![a-z0-9])" + k + r"(?![a-z0-9])", KB_LOWER):
            extra.append(ACRONYMS[k])
    extra += [rep for pat, rep in SYNONYMS if re.search(pat, ql)]
    return q + (" " + " ".join(extra) if extra else "")


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "es", "s", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def word_analyzer(text: str):
    toks = [stem(w) for w in re.findall(r"[a-z0-9]+(?:[./\-][a-z0-9]+)*", norm(text))]
    toks = [t for t in toks if t not in STOP]
    # also split compound tokens (http/3 -> http, 3 ; sha-256 -> sha, 256)
    extra = [p for t in toks if re.search(r"[./\-]", t) for p in re.split(r"[./\-]", t) if p and p not in STOP]
    toks = toks + extra
    bigrams = [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks + bigrams


def split_sentences(p: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", p) if s.strip()]


SENTS = [split_sentences(p) for p in PASSAGES]


def _build_acronyms() -> dict[str, str]:
    acr = {}
    for p in PASSAGES:
        words = re.findall(r"[A-Za-z][A-Za-z0-9]*", p.replace("-", " "))
        for n in range(2, 6):
            for i in range(len(words) - n + 1):
                chunk = words[i:i + n]
                if any(w.lower() in ("the", "of", "and", "a", "to", "in", "for", "by", "with") for w in chunk):
                    continue
                key = "".join(w[0] for w in chunk).lower()
                acr.setdefault(key, " ".join(chunk).lower())
    return acr


ACRONYMS = _build_acronyms()
TOPICS = [s[0] for s in SENTS]

# --------------------------------------------------------------------------- vectorizers
word_vec = TfidfVectorizer(analyzer=word_analyzer, sublinear_tf=True, norm="l2")
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                           preprocessor=norm, norm="l2")
corpus = PASSAGES + [s for ss in SENTS for s in ss]
word_vec.fit(corpus)
char_vec.fit(corpus)
P_WORD = word_vec.transform(PASSAGES)
P_CHAR = char_vec.transform(PASSAGES)
T_WORD = word_vec.transform(TOPICS)
T_CHAR = char_vec.transform(TOPICS)
FLAT_SENTS = [(i, j, s) for i, ss in enumerate(SENTS) for j, s in enumerate(ss)]
S_WORD = word_vec.transform([s for _, _, s in FLAT_SENTS])
S_CHAR = char_vec.transform([s for _, _, s in FLAT_SENTS])

# --------------------------------------------------------------------------- attributes
VAL_RE = {
    "port": re.compile(r"\bport\s+\d+", re.I),
    "duration": re.compile(r"\b\d+\s*ms\b", re.I),
    "ratio": re.compile(r"(?<![\d.])0\.\d+\b"),
    "bits": re.compile(r"\b\d+[ -]?bits?\b", re.I),
    "hash": re.compile(r"\b(SHA-256|SHA-3|BLAKE2|HMAC)\b"),
    "count": re.compile(r"\b(?:of|into|contains|stacks|organizes|across|minimum of)\s+\d+\b(?!\s*(?:ms|bits?))", re.I),
    "systems": re.compile(r"\b(?:[A-Z][A-Za-z0-9]+(?:\s+(?:and|&)\s+|,\s*|\s+)){1,}"
                          r"(?:implement|incorporate|rel(?:y|ies)|use|utilize|employ|popularized|support|"
                          r"default|demonstrate|depend|represent|mandates)"),
}

ATTR_PATTERNS = [
    ("port", r"\bports?\b|\blisten|\bbinds?\b"),
    ("bits", r"\bbits?\b|\bkey (?:length|size)s?\b|\bsecurity (?:level|strength)\b|\btag size\b"),
    ("hash", r"\bhash(?:ing)? (?:function|algorithm)|\bdigests?\b|\b(?:which|what) hash\b|\bprimitive\b|"
             r"\bhash (?:do|does)\b"),
    ("systems", r"\bwhich (?:systems?|databases?|products?|projects?|tools?|implementations?|companies|browsers?|"
                r"proxies|proxy|caches?|models?|services?|frameworks?|platforms?|curves?|vendors?|libraries)\b|"
                r"\bwho (?:uses?|implements?)\b|\bwhat (?:systems?|databases?|products?|software|tools?|projects?|"
                r"frameworks?|platforms?|curves?)\b|\breal[- ]world\b|\bexamples? of\b|\bimplemented (?:by|in)\b"),
    ("duration", r"\btime-?outs?\b|\btimes? out\b|\bintervals?\b|\bheartbeats?\b|\blatenc(?:y|ies)\b|\bms\b|"
                 r"\bmilliseconds?\b|\bhow long\b|\bduration\b|\b(?:grant|penalty|time) window\b|\blease\b|"
                 r"\bwait\b|\bdelays?\b|\btime limit\b|\bbudget\b|\bdrift\b|\bhow (?:quickly|fast|soon)\b|"
                 r"\bwithin how\b|\brto\b|\bkeepalive\b|\bexpire|\bteardown\b|\bclock skew\b"),
    ("ratio", r"\bthresholds?\b|\bratio\b|\bprobability\b|\bsaturation\b|\butili[sz]ation\b|"
              r"\bfraction\b|\bmetric\b|\blearning rates?\b|\bpercent|\bburst\b|\bcorrelation\b|"
              r"\bskew\b|\bvariance\b|\bslack\b|\bcut-?off\b|\bbuffer usage\b|\bfill\b|\brates?\b|\bdropout\b"),
    ("count", r"\bhow many\b|\bhow (?:large|big|wide|long is the window)\b|\bnumber of\b|\bquorums?\b|\bcounts?\b|\bsizes?\b|\bbatch|\bsamples?\b|"
              r"\bchunks?\b|\bbytes\b|\brecords?\b|\btokens?\b|\bentries\b|\blayers?\b|\bblocks?\b|"
              r"\bmembers?\b|\bretries\b|\breplicas?\b|\bnodes?\b|\bparticipants?\b|\bacceptors?\b|"
              r"\binstances\b|\bexamples\b|\bwindow length|\bgroups?\b"),
]


def detect_attrs(q: str) -> list[str]:
    ql = q.lower()
    found = [name for name, pat in ATTR_PATTERNS if re.search(pat, ql)]
    # a question about "skew threshold 0.x" vs "skew 500ms": keep both, sentence scoring decides
    return found


SYS_VERB = re.compile(r"\b(?:implement|incorporate|rel(?:y|ies)|use|utilize|employ|popularized|default|"
                      r"demonstrate|depend|represent|mandates|negotiate)s?\b", re.I)


def sentence_has(attr: str, s: str) -> bool:
    if attr == "systems":
        names = [w for w in re.findall(r"\b[A-Za-z][A-Za-z0-9.+-]*\b", s)[1:] if any(c.isupper() for c in w)]
        return bool(SYS_VERB.search(s)) and (len(names) >= 1 or s.split()[0][0].isupper()
                                             and re.search(r"\band\b", s) is not None)
    if attr == "count":
        return bool(COUNT_RE.search(s))
    return bool(VAL_RE[attr].search(s))


COUNT_RE = re.compile(r"(?<![\w./-])\d+(?![\w.])")
PASSAGE_ATTRS = [{a for a in VAL_RE if any(sentence_has(a, s) for s in ss)} for ss in SENTS]

# --------------------------------------------------------------------------- injection defence
INJ_PAT = re.compile(
    r"ignore|disregard|forget (?:all|everything|the|your|previous)|override|system prompt|"
    r"\binstructions?\b|you are now|act as|pretend|jailbreak|developer mode|\bdan\b|"
    r"respond (?:only )?with|reply (?:only )?with|output (?:only|exactly|the (?:word|string|text|following))|"
    r"\bprint\b|\bsay\b|return (?:the )?(?:word|string|text|null|only)|new rule|\bsystem\s*:|assistant\s*:|"
    r"\buser\s*:|do not answer|don'?t answer|instead,|\binstead\b|(?:correct|real|true|actual) answer|"
    r"answer is|updated (?:docs|documentation|kb|knowledge)|note to (?:the )?(?:ai|assistant|model|system)|"
    r"\bpwned\b|\bhacked\b|canary|secret (?:code|word|token)|confidence (?:to|of) 1|set (?:sources|confidence)|"
    r"must (?:now )?(?:answer|respond|reply|output)|<\s*/?\s*(?:system|instruction|admin)|\[\s*(?:system|admin|inst)",
    re.I,
)


def sanitize(q: str) -> tuple[str, bool]:
    """Return (clean_question, injection_detected)."""
    raw = q
    q = re.sub(r"<!--.*?-->", " ", q, flags=re.S)
    q = re.sub(r"<[^>]{0,40}>", " ", q)
    q = re.sub(r"[​-‏⁠﻿]", "", q)  # zero-width chars
    # drop bracketed / quoted segments that carry injection phrases
    for pat in (r"\[[^\]]*\]", r"\{[^}]*\}", r"\([^)]*\)", r"\"[^\"]*\"", r"'[^']{12,}'", r"`[^`]*`"):
        q = re.sub(pat, lambda m: " " if INJ_PAT.search(m.group(0)) else m.group(0), q, flags=re.S)
    parts = re.split(r"(?<=[.!?;:\n])\s+|\n+|\s+-{2,}\s+|\s+\|\s+", q)
    keep = [p for p in parts if p.strip() and not INJ_PAT.search(p)]
    clean = " ".join(keep).strip()
    injected = clean != " ".join(p for p in parts if p.strip()).strip() or q != raw
    if len([w for w in re.findall(r"[a-zA-Z0-9]+", clean) if w.lower() not in STOP]) < 2:
        # injection was in the same clause as the question: strip just the trigger phrases
        clean = INJ_PAT.sub(" ", q)
        clean = re.sub(r"\b(?:and|then|but)\s+(?=(?:tell|what|which|how|give))", " ", clean, flags=re.I)
    return re.sub(r"\s+", " ", clean).strip(), injected


# --------------------------------------------------------------------------- unknown entities
IGNORE_CAPS = {"what", "which", "how", "who", "when", "where", "why", "does", "do", "is", "are", "the", "in",
               "for", "according", "given", "i", "a", "an", "if", "on", "of", "and", "or", "can", "could",
               "please", "tell", "explain", "describe", "knowledge", "base", "passage", "context", "kb",
               "answer", "question", "note", "under", "with", "within", "by", "to", "at", "as", "from",
               "between", "compare", "both", "also", "list", "name", "state", "find", "give", "assuming",
               "suppose", "consider", "using", "used", "default", "standard", "typical", "true", "false"}


def _search_forms(tok: str) -> list[str]:
    t = re.sub(r"'s$", "", tok.lower().strip(".,;:?!'\"()[]"))
    forms = {t}
    if t.endswith("es") and len(t) > 4:
        forms.add(t[:-2])
    if t.endswith("s") and len(t) > 3:
        forms.add(t[:-1])
    if t in ALIASES:
        forms.add(ALIASES[t].replace(" " + t, "").strip())
    if tok.isupper() and len(t) >= 4 and t in ACRONYMS and not re.search(r"(?<![a-z0-9])" + re.escape(t), KB_LOWER):
        forms.add(ACRONYMS[t])
    return [f for f in forms if f]


def _find(form: str, text: str) -> int:
    m = re.search(r"(?<![a-z0-9])" + re.escape(form), text)
    return m.start() if m else -1


def kb_has_token(tok: str) -> bool:
    t = tok.lower().strip(".,;:?!'\"()[]")
    if not t or t in IGNORE_CAPS or len(t) < 2:
        return True
    if tok.isupper() and 4 <= len(t.rstrip("s")) <= 6 and (t in ACRONYMS or t.rstrip("s") in ACRONYMS):
        return True
    if any(_find(f, KB_LOWER) >= 0 for f in _search_forms(tok)):
        return True
    # hyphenated / slashed compounds: "Dynamo-style" -> known if its head part is known
    parts = [p for p in re.split(r"[-/+]", t) if len(p) >= 2]
    if len(parts) > 1 and not re.search(r"\d", t):
        return any(kb_has_token(p) for p in parts)
    return False


def named_tokens(q: str) -> list[str]:
    out = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9]*(?:[./\-+][A-Za-z0-9]+)*\+?", q):
        tok = m.group(0)
        prev = q[: m.start()].rstrip()
        sentence_initial = (not prev) or prev[-1] in ".!?:\n\"'(["
        has_digit = any(ch.isdigit() for ch in tok)
        has_alpha = any(ch.isalpha() for ch in tok)
        if has_digit and not has_alpha:
            if re.fullmatch(r"\d+\.\d+|\d{3,5}", tok):  # versions like 1.3, ports/codes like 429
                out.append(tok)
            continue
        if re.fullmatch(r"\d+(?:ms|s|bits?)", tok.lower()):
            continue
        all_caps = len(tok) >= 2 and tok.isupper()
        inner_caps = any(ch.isupper() for ch in tok[1:])
        capital = tok[0].isupper()
        if has_digit or all_caps or inner_caps or (capital and not sentence_initial):
            if tok.lower() not in IGNORE_CAPS:
                out.append(tok)
    # lowercase identifiers that are KB-specific names (etcd, mdns, ...)
    for w in re.findall(r"\b[a-z][a-z0-9]{2,}\b", q):
        if w in LOWER_NAMES and w not in [o.lower() for o in out]:
            out.append(w)
    return out


# lowercase tokens that are rare product/protocol names in the KB
LOWER_NAMES = {"etcd", "raft", "paxos", "swim", "saga", "sagas", "pgbouncer", "innodb", "mysql", "redis",
               "memcached", "mongodb", "cassandra", "rocksdb", "sqlite", "quic", "dns", "mdns", "bgp", "ospf",
               "tls", "tcp", "http", "jwt", "jwts", "pkce", "oauth", "aes", "rsa", "ecdsa", "chacha20",
               "argon2", "pbkdf2", "adamw", "sgd", "svm", "mlp", "resnet", "llama", "gpt", "riak", "dynamo",
               "consul", "zookeeper", "chubby", "temporal", "cadence", "squid", "envoy", "kong", "wireguard",
               "bonjour", "avahi", "wiredtiger", "postgres", "postgresql", "hlc", "wal", "lsm", "hpack",
               "qpack", "bbr", "dnssec", "ed25519", "convnext", "kafka", "zab", "etcd"}


def merge_phrases(q: str, names: list[str]) -> list[str]:
    """Join adjacent names ("Apache Cassandra") so one subject is boosted once."""
    out, i = [], 0
    while i < len(names):
        cur = names[i]
        while i + 1 < len(names) and re.search(re.escape(cur) + r"\s+" + re.escape(names[i + 1]), q):
            cand = cur + " " + names[i + 1]
            if not any(_find(f, KB_LOWER) >= 0 for f in [cand.lower()]):
                break
            cur = cand
            i += 1
        out.append(cur)
        i += 1
    return out


def unknown_entities(names: list[str]) -> list[str]:
    return [t for t in names if not kb_has_token(t)]


PASSAGE_LOWER = [p.lower() for p in PASSAGES]
SUBJECT_ZONE = [" ".join(p.lower().split()[:4]) for p in PASSAGES]


def entity_boost(names: list[str]) -> tuple[np.ndarray, dict]:
    boost = np.zeros(len(PASSAGES))
    hits = {}
    for tok in names:
        forms = _search_forms(tok)
        contains = [i for i, p in enumerate(PASSAGE_LOWER) if any(_find(f, p) >= 0 for f in forms)]
        if not contains or len(contains) > 12:
            continue
        hits[tok] = contains
        w = 0.09 / np.sqrt(len(contains))
        for i in contains:
            subj = any(_find(f, SUBJECT_ZONE[i]) >= 0 for f in forms)
            topic = any(_find(f, TOPICS[i].lower()) >= 0 for f in forms)
            boost[i] += w * (2.2 if subj else 1.4 if topic else 1.0)
    return boost, hits


# --------------------------------------------------------------------------- retrieval
def retrieve(q: str, names: list[str]):
    qn = expand_query(q)
    qw = word_vec.transform([qn])
    qc = char_vec.transform([qn])
    full_w = (P_WORD @ qw.T).toarray().ravel()
    full_c = (P_CHAR @ qc.T).toarray().ravel()
    top_w = (T_WORD @ qw.T).toarray().ravel()
    top_c = (T_CHAR @ qc.T).toarray().ravel()
    sw = (S_WORD @ qw.T).toarray().ravel()
    sc = (S_CHAR @ qc.T).toarray().ravel()
    sent_score = 0.6 * sw + 0.4 * sc
    best_sent = np.zeros(len(PASSAGES))
    for k, (i, _, _) in enumerate(FLAT_SENTS):
        best_sent[i] = max(best_sent[i], sent_score[k])
    attrs = detect_attrs(q)
    attr_bonus = np.array([0.04 * sum(a in pa for a in attrs[:2]) for pa in PASSAGE_ATTRS])
    eb, hits = entity_boost(merge_phrases(q, names))
    raw = 0.30 * full_w + 0.20 * full_c + 0.20 * top_w + 0.10 * top_c + 0.20 * best_sent + attr_bonus + eb
    return raw, sent_score, attrs, hits


def calibrate(raw: np.ndarray) -> np.ndarray:
    # map the hybrid score onto [0,1] (cosine-like). raw ~0.08 noise, ~0.25+ strong match
    return np.clip(1.0 / (1.0 + np.exp(-(raw - 0.16) * 18.0)), 0.0, 1.0)


STRONG = ("port", "bits", "hash")


def focus_scores(q: str, names: list[str]) -> np.ndarray:
    """Sentence scores for a query with the subject names removed (used inside a chosen passage)."""
    fq = q
    for n in sorted(names, key=len, reverse=True):
        fq = re.sub(r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![A-Za-z0-9])", " ", fq)
    content = [w for w in re.findall(r"[a-z0-9]+", fq.lower()) if w not in STOP]
    if len(content) < 1:
        fq = q
    fq = expand_query(fq)
    qw = word_vec.transform([fq])
    qc = char_vec.transform([fq])
    return 0.65 * (S_WORD @ qw.T).toarray().ravel() + 0.35 * (S_CHAR @ qc.T).toarray().ravel()


def best_answer(q: str, pid: int, names: list[str], attrs: list[str]) -> tuple[str | None, float]:
    """Pick the best supporting sentence inside passage pid (None = passage can't answer)."""
    subj = [n for n in names if any(_find(f, TOPICS[pid].lower()) >= 0 for f in _search_forms(n))]
    sent_score = focus_scores(q, subj)
    cands = [(k, j, s) for k, (i, j, s) in enumerate(FLAT_SENTS) if i == pid]
    present = [a for a in attrs if any(sentence_has(a, s) for _, _, s in cands)]
    strong_asked = [a for a in attrs if a in STRONG]
    if strong_asked and not any(a in present for a in strong_asked):
        return None, 0.0
    numeric_asked = [a for a in attrs if a in ("duration", "ratio", "count")]
    if numeric_asked and not any(a in present for a in numeric_asked) and "systems" not in present:
        return None, 0.0
    if present:  # only sentences that actually state the kind of value asked for
        cands = [c for c in cands if sentence_has(present[0], c[2])]
    scored = []
    for k, j, s in cands:
        sc = float(sent_score[k]) - (0.03 if j == 0 else 0.0)
        for a in present[1:3]:
            if sentence_has(a, s):
                sc += 0.08
        scored.append((sc, j, s))
    scored.sort(reverse=True)
    return scored[0][2], scored[0][0]


# --------------------------------------------------------------------------- API
app = FastAPI(title="RAG Colosseum")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["*"])

ABSTAIN_SCORE = float(os.environ.get("ABSTAIN_SCORE", "0.30"))
COMPOUND = re.compile(r"\b(?:and|both|versus|vs\.?|compared?|comparison|difference|differ|respectively|"
                      r"combined|together|sum|total|plus|each of)\b", re.I)


def abstain(reason: str, extra: dict | None = None):
    body = {"answer": None, "sources": [], "scores": [], "confidence": 0.05}
    if os.environ.get("DEBUG"):
        body["debug"] = {"reason": reason, **(extra or {})}
    return body


def answer_question(question: str, top_k: int = 3) -> dict:
    top_k = max(1, min(int(top_k or 3), 10))
    clean, injected = sanitize(question or "")
    if not clean:
        return abstain("empty")
    names = named_tokens(clean)
    unk = unknown_entities(names)
    raw, sent_score, attrs, hits = retrieve(clean, names)
    cal = calibrate(raw)
    order = [int(i) for i in np.argsort(-raw)]
    top = order[0]
    dbg = {"clean": clean, "names": names, "unknown": unk, "attrs": attrs, "raw_top": round(float(raw[top]), 4),
           "top": top, "injected": injected}
    if unk:
        return abstain("unknown entity", dbg)
    if cal[top] < ABSTAIN_SCORE:
        return abstain("low similarity", dbg)

    compound = bool(COMPOUND.search(clean))
    # passages to answer from: normally just the top one; for compound questions naming
    # several distinct subjects, the best passage for each subject
    targets = [top]
    if compound:
        for tok, ps in hits.items():
            if len(ps) <= 3:
                forms = _search_forms(tok)
                in_topic = [i for i in ps if any(_find(f, TOPICS[i].lower()) >= 0 for f in forms)]
                best = max(in_topic or ps, key=lambda i: raw[i])
                if best not in targets and cal[best] >= ABSTAIN_SCORE:
                    targets.append(best)
        if len(targets) == 1 and len(order) > 1:
            second = order[1]
            if raw[second] >= 0.75 * raw[top] and cal[second] >= ABSTAIN_SCORE:
                targets.append(second)
        targets = sorted(targets[:3], key=lambda i: -raw[i])

    answer_parts = []
    for pid in targets:
        ans, _ = best_answer(clean, pid, names, attrs)
        if ans is None:
            if pid == top and len(targets) == 1:
                return abstain("attribute not stated in passage", dbg)
            continue
        answer_parts.append(ans)
        # compound question asking two things about one subject: cover each attribute
        if compound and len(targets) == 1:
            for a in attrs[:3]:
                if not any(sentence_has(a, s) for s in answer_parts):
                    extra = [s for s in SENTS[pid] if sentence_has(a, s) and s not in answer_parts]
                    if extra:
                        fs = focus_scores(clean, names)
                        k0 = [k for k, (i, _, s) in enumerate(FLAT_SENTS) if i == pid and s in extra]
                        answer_parts.append(FLAT_SENTS[max(k0, key=lambda k: fs[k])][2])
    if not answer_parts:
        return abstain("attribute not stated in passage", dbg)

    ranked = sorted(set(order[:top_k]) | set(targets), key=lambda i: -raw[i])
    sources = ranked[: max(top_k, len(targets))]
    scores = [round(float(cal[i]), 4) for i in sources]
    margin = float(raw[top] - raw[order[1]]) if len(order) > 1 else 0.2
    confidence = round(float(np.clip(0.62 + 0.25 * cal[top] + 0.6 * margin, 0.61, 0.99)), 4)
    body = {"answer": " ".join(answer_parts), "sources": sources, "scores": scores, "confidence": confidence}
    if os.environ.get("DEBUG"):
        body["debug"] = dbg
    return body


@app.get("/")
def root():
    return {"status": "ok", "passages": len(PASSAGES), "endpoint": "POST /query"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
async def query(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    q = payload.get("question") or payload.get("query") or payload.get("q") or ""
    top_k = payload.get("top_k", 3)
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 3
    body = answer_question(str(q), top_k)
    return JSONResponse(body, headers={"Access-Control-Allow-Origin": "*"})


@app.post("/query/")
async def query_slash(request: Request):
    return await query(request)
