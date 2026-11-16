# RAG Colosseum service

FastAPI RAG endpoint over the 50-passage `knowledge_base.json`.

- `POST /query` `{"question": "...", "top_k": 3}` → `{"answer", "sources", "scores", "confidence"}`
- `GET /health` for Render's health check
- CORS `Access-Control-Allow-Origin: *` on every response

## How it works

Retrieval is a hybrid lexical-semantic ranker (word TF-IDF + bigrams, char 3–5-gram TF-IDF,
topic-sentence match, IDF-weighted entity/subject boost, acronym expansion built from the KB, and an
attribute boost so "what port…" favours passages that state a port).

Answers are extracted verbatim from the best passage (the sentence that states the asked-for kind of
value), so injected instructions in a question can never appear in the answer.

It abstains (`answer: null`, `sources: []`, `confidence: 0.05`) when:
- the question names an entity that is not in the KB (Kafka, Zab, RMSProp, Twofish…),
- the matched passage does not state the kind of value asked for (e.g. a port for etcd), or
- similarity is low (off-topic).

## Deploy on Render

New → Web Service → connect this repo. Render reads `render.yaml`:
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Test locally

```bash
pip install -r requirements.txt
python tests/test_probe.py && python tests/test_heldout.py && python tests/test_hard.py
uvicorn app:app --port 8000
```

Set `DEBUG=1` to include retrieval diagnostics in responses.
