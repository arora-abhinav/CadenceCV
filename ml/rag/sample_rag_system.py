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

    #intake - what the runner told us, NOT measured from video. this is what drives the report's resolution
    #hierarchy: anterior knee pain + a 45% weekly mileage jump should make it lead with the volume spike,
    #target cadence (a load-lowering change), and refuse the forefoot transition (loads the painful area more)
    "intake": {
        "pain": {
            "location": "knee_anterior",
            "side": "left",
            "duration": "3 weeks",
            "aggravated_by": "downhill",
        },
        "weekly_km_change_pct": 45,
    },
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

#Overriding the metrics for now
with open("/Users/abhinavarora/Desktop/CadenceCV/evaluate_metrics.json", "r") as file:
    metrics = json.load(file)

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


OUT = Path("ml/rag/rag artifcats/queries.json")

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

-Note: if any metrics are blank (NaN values, None values, empty arrays), simply ignore reporting them
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


def _load_doc_headers():
    #pmcid -> a short provenance header i staple onto each chunk so the report model can weight the evidence
    #(author/year/journal are always filled; design/n_participants/population are backfill fields, still all
    #null for now). i only emit the parts that actually exist, so once i backfill design/n/pop they show up
    #here automatically and i never feed the model a useless 'design: None'.
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT pmcid, first_author, year, journal, design, n_participants, population FROM documents"
        )).fetchall()
    headers = {}
    for pmcid, author, year, journal, design, n, pop in rows:
        bits = []
        if author:
            bits.append(f"{author} ({year})" if year else author)
        if journal:
            bits.append(journal)
        if design:
            bits.append(design)
        if n:
            bits.append(f"n={n}")
        if pop:
            bits.append(pop)
        headers[pmcid] = "[" + " | ".join(bits) + "]"
    return headers


def dense_retrieve(passages, top_k=10):
    #pull every chunk's stored vector + text back out of pgvector. the embedding comes across as pgvector's
    #text form '[0.1, 0.2, ...]', which is already valid json, so json.loads -> np array rebuilds the vector
    doc_headers = _load_doc_headers()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT embedding, text, pmcid FROM chunks")).fetchall()

    corpus_embs = np.array([json.loads(r[0]) for r in rows], dtype=np.float32)
    #staple the provenance header on. the embedding was made from the raw text (header doesnt touch ranking),
    #this only changes what text gets RETURNED to the report model
    corpus_texts = [f"{doc_headers.get(r[2], '')}\n{r[1]}".strip() for r in rows]

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
    doc_headers = _load_doc_headers()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT text, pmcid FROM chunks")).fetchall()

    #two views of the corpus: raw content for BM25 (i DONT want author names/years polluting the lexical
    #index), and a header-stapled version thats what actually gets returned to the report model
    raw_texts = [r[0] for r in rows]
    corpus_texts = [f"{doc_headers.get(r[1], '')}\n{r[0]}".strip() for r in rows]
    #bm25 wants the corpus pre-tokenized up front; it builds the term stats once and scores against that
    bm25 = BM25Okapi([_tokenize(t) for t in raw_texts])

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


#The report writer. This is the "stronger LLM" stage - it gets the measured numbers + the retrieved
#literature and nothing else, and has to reason under all the clinical guardrails below.
REPORT_SYSTEM = """You are generating a running gait analysis report. You are the analysis
component of a video-based tool — you have measured numbers and research
literature, and nothing else.

WHAT YOU CANNOT DO
You did not examine this runner. No strength testing, no range of motion,
no palpation, no watching them move in person, no follow-up. A physiotherapist
resolves ambiguity with an exam; you resolve it with the intake below or you
acknowledge you can't. Never write as though you observed anything beyond
what is in MEASURED.

RESOLVING CONFLICTING EVIDENCE
The retrieved literature will contradict itself. This is expected — it
reflects the actual state of the field. Do not present both sides neutrally
and leave the runner to decide. Resolve it for THIS runner, in this order:

1. Pain present → conservative. A change that increases load on the painful
   structure loses to a general benefit, every time.
2. Recent training volume increase → that is the likely driver. Say so before
   discussing gait.
3. No pain, performance goal → economy literature applies; you can be more
   assertive.
4. Neither pain nor goal → describe what you measured, prescribe nothing.

READING THE EVIDENCE
Each chunk carries a header with author, year, design, sample size, and
population. Use it.

- Distinguish HABITUAL from IMPOSED from RETRAINED. A study of runners who
  naturally run a certain way says something different from one that
  instructed a change for ten minutes, which says something different again
  from twelve weeks of retraining. Never merge them.
- Distinguish the specific variable. "Peak knee flexion" is not "knee flexion
  at initial contact." "Vertical oscillation" is not "vertical oscillation
  normalized to step length." If the evidence is about a neighbouring
  variable, either say so or don't use it.
- Weight by design and sample size in your wording: "one study of eleven
  runners found" vs "a meta-analysis of 51 studies found." Never "research
  shows."
- Some chunks are methods sections describing apparatus and protocol. They
  contain no findings. Ignore them.
- Two papers may share a cohort. Do not treat them as independent
  corroboration.
- Running speed determines nearly every metric. Evidence collected at a
  different speed than this runner's transfers weakly. Say so when it matters.

WRITING THE REPORT

Open with their complaint or goal, not the metric list.

Give ONE primary recommendation. Two maximum. A report with five
recommendations is a report the runner will act on none of.

Say nothing about a metric where the corpus gives you no basis to. Silence
is correct; filler is not. If a displayed value is null it was suppressed
for capture-quality reasons — do not mention it at all.

Be decisive but calibrated. Not "you may wish to consider possibly" — but
also not certainty you don't have. A clinician says "let's try this and see
how you respond," which is hedged without being evasive. Aim there.

If pain is reported: describe what the data shows, recommend they see a
physiotherapist or sports medicine doctor, and do NOT name a diagnosis or
clear them to keep running.

400-600 words. Plain language — a runner reads this, not a biomechanist.
Define any technical term you use once.

STRUCTURE
  What you told us     — 1-2 sentences, complaint/goal and training context
  What we measured     — the notable values only, with speed as context
  What this suggests   — your reasoning, with attributed evidence
  What to do           — one or two actions, with a way to check if it worked

Do not invent numbers. Every value comes from MEASURED; every claim comes
from a chunk you were given."""


def generate_report(metrics, dense, sparse):
    #gather every chunk both retrievers pulled, deduped by text (a chunk often lands in BOTH the dense and
    #sparse top-10). skipping RRF for now - the model just gets the raw union of what we retrieved.
    seen = {}
    for retrieved in (dense, sparse):
        for matches in retrieved.values():
            for m in matches:
                seen[m["text"]] = m["text"]
    corpus = list(seen.values())

    #the model gets exactly two things: the measured numbers + the retrieved literature (per the system prompt)
    user_content = json.dumps(
        {"MEASURED": metrics, "RETRIEVED_CHUNKS": corpus},
        indent=2, ensure_ascii=False,
    )

    resp = client.messages.create(
        model="claude-sonnet-5",  #stronger model for the actual reasoning/writing step
        max_tokens=10000,
        system=REPORT_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    #sonnet 5 returns an extended-thinking block FIRST, so content[0] isnt the answer. grab the text
    #block(s) explicitly and ignore the thinking - join in case the reply comes in more than one text block
    report = "".join(block.text for block in resp.content if block.type == "text")
    Path("ml/rag/rag artifcats/report.md").write_text(report)
    print(f"\n########## GAIT REPORT ({len(corpus)} chunks in context) ##########\n")
    print(report)
    return report


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
    Path("ml/rag/rag artifcats/retrieved_dense.json").write_text(json.dumps(dense, indent=2, ensure_ascii=False))
    Path("ml/rag/rag artifcats/retrieved_sparse.json").write_text(json.dumps(sparse, indent=2, ensure_ascii=False))

    for label, retrieved in [("DENSE (cosine)", dense), ("SPARSE (bm25)", sparse)]:
        print(f"\n########## {label} ##########")
        for metric_id, matches in retrieved.items():
            print(f"\n=== top {len(matches)} chunks for [{metric_id}] ===")
            for rank, m in enumerate(matches, 1):
                print(f"{rank:2d}. {m['score']:.4f}  {m['text'][:110].replace(chr(10), ' ')}...")
    print("\nwrote retrieval results to retrieved_dense.json and retrieved_sparse.json")

    #hand the measured numbers + the retrieved chunks to sonnet to write the actual report
    generate_report(metrics, dense, sparse)

    return passages

if __name__ == "__main__":
    passages = main()