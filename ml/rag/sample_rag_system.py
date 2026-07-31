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
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

OUT = Path("queries.json")

SYSTEM = """You convert running gait metrics into retrieval passages. Each passage is
embedded and matched against chunks of real biomechanics papers, so it must
read like paper text — not like a question.

RULES
- Declarative sentences only. Never write a question.
- Use the literature's vocabulary, not the display name. Work the provided
  aliases in naturally.
- State relationships in BOTH directions. Never assert that higher or lower
  is better — that biases retrieval toward papers that already agree.
- Name the outcomes the metric is studied against: running economy, injury
  risk, joint loading, ground reaction forces, performance.
- Do not mention any runner, any measured value, or any advice.
- 2-3 sentences, 40-60 words.

EXAMPLE
input:  {"metric_id": "cadence", "aliases": ["step rate", "stride frequency", "SPM"]}
output: {"metric_id": "cadence", "passage": "Step rate, also reported as cadence or stride frequency, was measured in recreational runners during treadmill and overground running. Changes in step rate have been associated with step length, vertical excursion of the centre of mass, braking impulse, and joint loading at the knee and hip. Lower step rates have been examined in relation to running economy and injury risk."}

OUTPUT
Raw JSON array only. No markdown fences, no preamble. One element per input
metric, same order:
[{"metric_id": "...", "passage": "..."}]"""

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


if __name__ == "__main__":
    main()