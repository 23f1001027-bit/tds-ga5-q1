import os, re
os.environ["DEBUG"] = "1"
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import answer_question
CASES = [
    ("What learning rate does the adaptive optimizer with decoupled weight decay start at?", 11, ["0.001"]),
    ("Which optimizer with momentum uses a base learning rate, and what is it?", 10, ["0.1"]),
    ("What quorum does the consensus protocol with 100ms heartbeats and log compaction require?", 0, ["5"]),
    ("In the leaderless consensus family used by Chubby, how many acceptors form the commit quorum?", 1, ["4"]),
    ("Which membership protocol uses indirect ping probes, and what is its eviction threshold?", 3, ["0.85"]),
    ("What threshold marks a node dead under the phi accrual detector?", 2, ["0.5"]),
    ("For the relational database that uses undo log segments, what port does it listen on?", 21, ["5432"]),
    ("For the relational database that uses page-level MVCC and vacuum, what port does it listen on?", 20, ["9200"]),
    ("What is the timeout before compensating actions run for long-running microservice transactions?", 5, ["1000ms"]),
    ("What hash algorithm does the key derivation function using HMAC iterations rely on?", 43, ["SHA-3"]),
    ("What output does the logistic regression classifier threshold at?", 14, ["0.7"]),
    ("Which hashing scheme picks the highest scoring node instead of using a ring, and what threshold scales weights?", 7, ["0.85"]),
    ("What does the system print to the log before dirty buffer flushes in PostgreSQL?", 20, ["write-ahead"]),
    ("Which stream cipher avoids table lookups and what key size does it use?", 47, ["512"]),
    ("What size groups does the TLS version with a two round trip handshake use for Diffie-Hellman?", 37, ["128"]),
    ("What is the security level of Diffie-Hellman in the single round trip TLS handshake?", 36, ["192"]),
    ("Which in-memory store without persistence listens on port 6379?", 27, ["6379"]),
    ("What port does the store with RDB snapshots and AOF logs use?", 26, ["27017"]),
    ("How many tokens per gradient step does the GPT style model train on?", 13, ["32"]),
    ("How many blocks does the bidirectional encoder backbone contain?", 12, ["4"]),
    ("What is the lock wait timeout in MySQL and the idle timeout in PgBouncer?", (21, 20), ["1000ms", "500ms"]),
    ("What are the ports used by MongoDB and Memcached?", (24, 27), ["27017", "6379"]),
]
f = 0
for q, exp, subs in CASES:
    r = answer_question(q, 3); a = r["answer"] or ""
    exps = exp if isinstance(exp, tuple) else (exp,)
    ok = bool(r["sources"]) and r["sources"][0] in exps and all(e in r["sources"] for e in exps) and all(s.lower() in a.lower() for s in subs)
    if not ok:
        f += 1; print("FAIL", exp, subs, "\n  Q:", q, "\n  ->", r["sources"], r["scores"], "\n  A:", a, "\n  ", r.get("debug"), "\n")
print(f"{len(CASES)-f}/{len(CASES)}")
