#This is a sample rag system that tests the entire test workflow now:
# 1) Running metrics through a claude Haiku model and obtaining proper queries
# 2) Obtaining all chunks based on their encoded data
# 3) Running a dense encoder on all chunks
# 4) Filtering further using the documents db
# 5) Submitting chunks as well as queries to a stronger LLM for analysis
metrics = {
    "Running Speed (m/s)": 3.00,

    "Cadence (steps/min)": 168.0,
    "Contact Time (s)": 0.245,
    "Flight Time (s)": 0.112,
    "Duty Factor (%)": 34.3,

    "Stride Length (m)": 2.143,
    "Vertical Oscillation (cm)": 8.6,
    "Vertical Ratio (%)": 8.03,
    "Left Overstride (% leg length)": 17.2,
    "Right Overstride (% leg length)": 18.0,

    "Left Foot Strike Angle (deg)": 11.4,
    "Right Foot Strike Angle (deg)": 12.8,
    "Left Knee Flexion at Contact (deg)": 17.2,
    "Right Knee Flexion at Contact (deg)": 16.5,
    "Trunk Lean (deg)": 6.8,
}


import json
import os
import re
from pathlib import Path

import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sqlalchemy import create_engine, text

load_dotenv(override=True)
API_KEY = os.environ["ANTHROPIC_API_KEY"]
client = Anthropic(api_key=API_KEY)

#Same postgres/pgvector db the chunks got written into
engine = create_engine(os.getenv("DATABASE_URL"))

#The SAME encoder that made the chunk embeddings - has to match or the query/chunk vectors live in
#different spaces and cosine is meaningless. Loading the presaved copy so i dont redownload weights.
transformer_path = os.path.join(os.path.dirname(__file__), "transformer_models", "sentence_transformer.pkl")
if os.path.exists(transformer_path):
    encoder = SentenceTransformer(transformer_path)
else:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


OUT = Path("queries.json")

SYSTEM = """
Write 2-3 sentences in the register of a DISCUSSION section of a
biomechanics paper — where findings are interpreted, not where methods
are described.

- Use interpretive vocabulary: associations, findings, evidence, magnitude,
  correlations, mechanisms, inconsistent, trivial, moderate.
- Do NOT describe apparatus, sampling rates, marker sets, participant
  counts, or protocols. No "participants ran on a treadmill at X m/s."
- State that relationships have been EXAMINED and that findings have
  VARIED. Never state a direction or a verdict.
- Do not invent statistics.

OUTPUT
Raw JSON array only. No markdown fences, no preamble. One element per input
metric, same order:
[{"metric_id": "...", "passage": "..."}]

- Report that a relationship WAS EXAMINED. Never state what was found.
  Yes: "Associations between step rate and knee joint loading were examined
       across a range of imposed cadences."
  No:  "Increasing step rate reduced knee loading."
- Do not invent statistics, correlation coefficients, p-values, or sample sizes.
  Protocol vocabulary only: population type, setting, speed range.
  """

ONTOLOGY = [
    {"metric_id": "cadence",
     "aliases": ["step rate", "stride frequency", "steps per minute"]},
    {"metric_id": "contact_time",
     "aliases": ["ground contact time", "stance time", "contact phase duration"]},
    {"metric_id": "flight_time",
     "aliases": ["aerial time", "flight phase duration", "non-contact time"]},
    {"metric_id": "duty_factor",
     "aliases": ["duty factor", "ratio of contact time to stride time"]},
    {"metric_id": "stride_length",
     "aliases": ["stride length", "step length", "stride amplitude"]},
    {"metric_id": "vertical_oscillation",
     "aliases": ["vertical oscillation", "centre of mass vertical displacement",
                 "vertical excursion"]},
    {"metric_id": "vertical_ratio",
     "aliases": ["vertical ratio", "vertical oscillation normalised to step length"]},
    {"metric_id": "overstride",
     "aliases": ["overstriding", "foot landing position relative to the centre of mass",
                 "braking force"]},
    {"metric_id": "foot_strike_angle",
     "aliases": ["foot strike angle", "foot inclination angle", "foot contact angle",
                 "rearfoot forefoot strike pattern"]},
    {"metric_id": "knee_flexion_contact",
     "aliases": ["knee flexion angle at initial contact",
                 "sagittal knee kinematics at footstrike"]},
    {"metric_id": "trunk_lean",
     "aliases": ["trunk lean", "trunk flexion angle", "forward postural lean",
                 "trunk inclination"]},
]


def dense_retrieve(passages, top_k=10):
    #pull every chunk's stored vector + text back out of pgvector. the embedding comes across as pgvector's
    #text form '[0.1, 0.2, ...]', which is already valid json, so json.loads -> np array rebuilds the vector
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT embedding, text FROM chunks")).fetchall()

    corpus_embs = np.array([json.loads(r[0]) for r in rows], dtype=np.float32)
    corpus_texts = [r[1] for r in rows]

    #embed each haiku passage with the same encoder that built the chunk vectors
    query_embs = encoder.encode([p["passage"] for p in passages])

    #cosine top-k per query. semantic_search gives back, per query, a ranked list of {corpus_id, score}
    hits = util.semantic_search(query_embs, corpus_embs, top_k=top_k)

    #re-key the ranked hits by metric so i can see which chunks each metric's passage pulled
    results = {}
    for p, query_hits in zip(passages, hits):
        results[p["metric_id"]] = [
            {"score": round(h["score"], 4), "text": corpus_texts[h["corpus_id"]]}
            for h in query_hits
        ]
    return results


def _tokenize(s):
    #dead simple lexical tokens for bm25 - lowercase, split on anything non-alphanumeric. no stemming/stopwords,
    #the corpus is tiny so its not worth it, and keeping units/numbers as tokens actually helps here
    return re.findall(r"[a-z0-9]+", s.lower())


def _tanh_normalise(scores):
    #squash bm25 scores through tanh so they land on the same (-1, 1) scale as the dense cosine scores.
    #catch: raw bm25 values (~3-20) sit way past tanhs sensitive zone and would ALL saturate to ~1.0, wiping
    #out the ranking. so i scale each querys scores by its own max first (-> [0,1]), THEN tanh - keeps them in
    #the responsive part of the curve and order is untouched (tanh is monotonic).
    scores = np.asarray(scores, dtype=np.float32)
    peak = scores.max()
    if peak > 0:
        scores = scores / peak
    return np.tanh(scores)


def sparse_retrieve(passages, top_k=10):
    #BM25 lexical search - the sparse counterpart to the dense encoder. this matches on ACTUAL shared words
    #(great for exact terms + aliases the embedding tends to smear together), no vectors involved at all.
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT text FROM chunks")).fetchall()

    corpus_texts = [r[0] for r in rows]
    #bm25 wants the corpus pre-tokenized up front; it builds the term stats once and scores against that
    bm25 = BM25Okapi([_tokenize(t) for t in corpus_texts])

    results = {}
    for p in passages:
        #get_scores returns one bm25 score per corpus doc; tanh-squash onto the dense (-1,1) scale, then
        #argsort desc for the top_k (squashing doesnt change the order, its just for a fair-looking score)
        scores = _tanh_normalise(bm25.get_scores(_tokenize(p["passage"])))
        top_idx = np.argsort(scores)[::-1][:top_k]
        results[p["metric_id"]] = [
            {"score": round(float(scores[i]), 4), "text": corpus_texts[i]}
            for i in top_idx
        ]
    return results


def main():
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=3000,
        system=SYSTEM,
        messages=[
            {"role": "user", "content": json.dumps({"metrics": ONTOLOGY})},
            {"role": "assistant", "content": "["},  # prefill: forces raw JSON
        ],
    )

    raw = "[" + resp.content[0].text
    passages = json.loads(raw)

    # guard against hallucinated or dropped metric_ids
    want = {m["metric_id"] for m in ONTOLOGY}
    got = {p["metric_id"] for p in passages}
    if want != got:
        raise ValueError(f"missing: {want - got}  unexpected: {got - want}")

    OUT.write_text(json.dumps(passages, indent=2, ensure_ascii=False))

    for p in passages:
        n = len(p["passage"].split())
        flag = "" if 35 <= n <= 70 else "   <-- length off, check this one"
        print(f"\n[{p['metric_id']}] ({n} words){flag}\n{p['passage']}")

    print(f"\nwrote {len(passages)} passages to {OUT}")

    #two retrievers running side by side over the same passages, each kept in its OWN file so i can
    #eyeball what dense (semantic) vs sparse (lexical) actually pull for each metric before i fuse them
    dense = dense_retrieve(passages, top_k=10)
    sparse = sparse_retrieve(passages, top_k=10)
    Path("retrieved_dense.json").write_text(json.dumps(dense, indent=2, ensure_ascii=False))
    Path("retrieved_sparse.json").write_text(json.dumps(sparse, indent=2, ensure_ascii=False))

    for label, retrieved in [("DENSE (cosine)", dense), ("SPARSE (bm25)", sparse)]:
        print(f"\n########## {label} ##########")
        for metric_id, matches in retrieved.items():
            print(f"\n=== top {len(matches)} chunks for [{metric_id}] ===")
            for rank, m in enumerate(matches, 1):
                print(f"{rank:2d}. {m['score']:.4f}  {m['text'][:110].replace(chr(10), ' ')}...")
    print("\nwrote retrieval results to retrieved_dense.json and retrieved_sparse.json")

    return passages

if __name__ == "__main__":
    passages = main()