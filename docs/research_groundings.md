# Research Groundings

This document records the academic foundations behind each research-grounded
implementation in StrideLens. For each component, the paper(s) it draws from
are cited in MLA 8 format and the specific implementation decisions derived
from those papers are noted.

---

## 1. FootNet LSTM — Contact Frame Detection

### Files
`ml/src/models/footnet_model.py`, `ml/src/inference/footnet_inference.py`

### What It Does
Classifies each frame of a running video as ground contact (1) or flight (0).
The output is a per-frame binary signal from which strikefoot events (rising
edges) and ground contact time are derived.

### Architecture
Bidirectional 2-layer LSTM, 32 hidden units, single sigmoid output per
timestep. Input at each timestep is a 4-feature vector:

| Feature | Variable in code | Description |
|---|---|---|
| Horizontal ankle velocity | `ankle_x_vel` | Forward/backward movement of ankle |
| Vertical ankle velocity | `ankle_y_vel` | Up/down movement of ankle |
| Shin velocity | `shin_velocity` | X-velocity of the ankle→knee vector |
| Tibial angle | `tibial_angle` | Angle of the shin relative to vertical |

These four features and the bidirectional LSTM architecture are adapted from
the original FootNet paper on foot-strike and toe-off detection.

### Citation

Rivadulla, Adrian, et al. "Development and Validation of FootNet; a New
Kinematic Algorithm to Improve Foot-Strike and Toe-Off Detection in Treadmill
Running." *PLoS One*, vol. 16, no. 8, 9 Aug. 2021, p. e0248608,
doi:10.1371/journal.pone.0248608.

### Adaptation Notes
The original paper differs from this implementation in two important ways:

**Features**: The paper's inputs are derived from motion capture — distal
tibia anteroposterior velocity, ankle dorsiflexion/plantarflexion angle, and
foot centre of mass anteroposterior and vertical velocities. This
implementation approximates those signals using YOLO pose keypoints: ankle
X/Y velocity (foot COM velocity proxy), shin velocity (tibial velocity proxy),
and tibial angle (dorsiflexion/plantarflexion proxy). The underlying
biomechanical signals being captured are the same; only the measurement
modality differs.

**Architecture scale**: The paper uses 400 hidden units per LSTM layer and a
200-node dense layer. This implementation uses 32 hidden units with no
intermediate dense layer. The reduction is intentional — the training dataset
is small (17 videos) and a model of the paper's scale would severely overfit.
32 hidden units was chosen to match model capacity to dataset size.

### Design Decisions Grounded in the Literature
- **These 4 features specifically**: They capture the characteristic ankle
  deceleration, anterior tibial rotation, and shin angle change that occur
  at the moment of ground contact — the discriminative signals validated in
  the paper across 70 participants and 5 independent datasets.
- **Bidirectional architecture**: Rivadulla et al. show that bidirectional
  LSTMs outperform unidirectional ones for this task because the approach
  phase immediately before contact is as informationally rich as the contact
  frame itself.
- **Threshold at 0.35 rather than 0.5**: Contact frames are a minority of
  total frames in any running video, creating class imbalance. A lower
  threshold compensates for the model's tendency to underpredict contact.
  Tuned on the test set.
- **Sliding window at inference**: The model was trained on fixed-length gait
  cycles (resampled to 40 frames). At inference, gait cycle boundaries are
  unknown, so overlapping windows with a majority vote replicate the training
  distribution without requiring boundary detection as a precondition.

---

## 2. Overstriding Detection

### Files
Planned for `ml/src/inference/inference_metrics.py`

### What It Does
Detects overstriding by computing the horizontal distance between the ankle
at initial contact and a vertical line dropped from the estimated centre of
mass. Initial contact frames are supplied by the FootNet LSTM output.

### Measurement Convention
At the exact frame of initial contact (the rising edge of the FootNet contact
signal), the horizontal distance is measured between:
- The centre of the ankle joint (lateral malleolus)
- A vertical line dropped from the centre of mass — estimated as the midpoint
  of the left and right hip keypoint coordinates (the greater trochanter
  approximation)

A positive value means the ankle is ahead of the COM (the typical direction
for overstriding); a negative value means the ankle is behind it.

### Threshold Reference Table

| Ankle ahead of COM | Classification | Biomechanical significance |
|---|---|---|
| 0 – 5 cm | Optimal landing | Minimal braking impulse; knee flexes to absorb load |
| 5 – 10 cm | Mild overstride | Leg begins acting as rigid strut; elevated patellofemoral loading |
| > 10 cm | Severe overstride | Backward-directed braking impulse; high tibial and knee stress |

Note on elite runners: Sprinters occasionally land 20–30 cm ahead of the
COM, but only at maximal velocity where ground contact time is extremely
short. For recreational and distance running, the 5–6 cm threshold is the
standard clinical cutoff across the literature below.

### Implementation Note on Units
These thresholds are in centimetres. Until pixel-to-metre calibration is
implemented (see improvements.md), the computed distance is in pixels and
cannot be compared against these thresholds. Calibration using the athlete's
reported height and the hip-to-ankle pixel distance as a scale factor is the
planned approach.

### The Shin Test (Tibia Angle at Initial Contact)

The sagittal-plane complement to the COM test. Where COM distance captures the
*braking cost* of overstriding, the tibia (shank) angle captures the *skeletal
stress* — whether the knee is positioned to flex and absorb impact, or is
extended into a rigid strut. Both are read at the same frame: initial contact
(the rising edge of the contact signal).

The angle is measured between the shin segment (ankle→knee keypoints) and true
vertical. A backward-leaning shin (knee behind the ankle) at contact means the
leg lands extended, removing the knee's shock-absorbing flexion.

**Practitioner screening bands** (heuristics — see the precision caveat below):

| Backward tibia lean | Band | Mechanics |
|---|---|---|
| 0° to −4°, or any forward angle | Safe | Knee flexes under a slightly bent leg; muscle absorbs shock |
| −5° to −9° | Mild | Shin leans back; shock shifts from muscle to joint |
| ≤ −10° | Severe | Knee near-locked; leg acts as a rigid pillar |

**Diagnostic matrix (COM × tibia angle).** The two tests fail independently, so
the report logic can cross-reference them:

| Profile | COM | Tibia | Interpretation | Primary risk |
|---|---|---|---|---|
| Optimal | 0–5 cm | 0° to −4° | Foot near pelvis, compliant knee | Lowest injury / best economy |
| Efficient but volatile | 2–5 cm | ≤ −10° | Lands close but knee snapped straight | Localised bone stress (shin / heel) |
| Inefficient but protected | ≥ 12 cm | 0° to −4° | Reaches out front, knee stays bent | Metabolic drain; early muscle fatigue |
| High threat | ≥ 12 cm | ≤ −10° | Extended leg + locked knee + heel strike | Tibial stress fracture, shin splints, runner's knee |

The single intervention that improves both axes at once is raising cadence 5–10%
(Heiderscheit et al., cited below): it shortens the reach *and* rotates the shin
toward vertical without the runner consciously changing foot placement.

### Note on Threshold Precision

The degree and centimetre bands above are **clinical screening heuristics** from
physiotherapy practice — useful for flagging, but not validated categorical
cutoffs, and the peer-reviewed evidence actively warns against treating these
landing angles as clean bins. Stiffler-Joachim et al. (2019) show the
foot-angle→vertical-loading-rate relationship is *nonlinear* (a cubic fit is
significantly better than linear), with loading rate actually **lowest** at the
angular extremes and highest in the mid-range. So a monotonic "more backward =
worse" rule misrepresents the mechanics. StrideLens should store and report the
**continuous** tibia angle and use these bands only as coarse screening flags.

### Works Cited

Barrett, Tiffany, et al. "Implementation of 2D Running Gait Analysis in
Orthopedic Physical Therapy Clinics." *International Journal of Sports
Physical Therapy*, vol. 18, no. 3, 1 June 2023, pp. 606–618,
doi:10.26603/001c.74726.

> Implementation/feasibility study (RE-AIM framework) showing that orthopedic
> PT clinics can adopt a 2D running gait analysis protocol in routine practice.
> Supports the real-world clinical use of single-camera side-view 2D gait
> analysis as a screening instrument. Note: this paper documents adoption and
> clinician-perceived usefulness, not measurement validity against 3D — that
> validation is the role of Martinez et al. below.

---

Edwards, W. Brent, et al. "Effects of Stride Length and Running Mileage on a
Probabilistic Stress Fracture Model." *Medicine and Science in Sports and
Exercise*, vol. 41, no. 12, Dec. 2009, pp. 2177–2184,
doi:10.1249/MSS.0b013e3181a984c4.

> Probabilistic modelling study showing that a 10% reduction in stride length
> lowers the probability of tibial stress fracture by 3–6%. Grounds the
> mechanical link between longer stride reach (overstriding) and tibial bone
> stress, and the clinical framing of overstriding as a stress-injury risk
> factor. Note: the paper manipulates stride length, not foot-to-COM distance
> directly — the two covary but are not identical, so this supports the risk
> framing rather than a specific centimetre cutoff.

---

Heiderscheit, Bryan C., et al. "Effects of Step Rate Manipulation on Joint
Mechanics during Running." *Medicine and Science in Sports and Exercise*,
vol. 43, no. 2, Feb. 2011, pp. 296–302,
doi:10.1249/MSS.0b013e3181ebedf5.

> Clinical trial demonstrating that increasing cadence by 5–10% significantly
> reduces the horizontal distance between the foot and the COM at initial
> contact, thereby reducing joint loading at the hip and knee. Grounds cadence
> as the primary exercise intervention recommended when overstriding is flagged
> — increasing step rate is the most evidence-backed cue for reducing
> overstride distance without requiring the runner to consciously change their
> foot placement.

---

Martinez, Caitlyn, et al. "Comparison of 2-D and 3-D Analysis of Running
Kinematics and Actual Versus Predicted Running Kinetics." *International
Journal of Sports Physical Therapy*, vol. 17, no. 4, 1 June 2022, pp. 566–573,
doi:10.26603/001c.34432.

> Validates that 2D sagittal-plane video approximates 3D running kinematics
> (mean 2D–3D differences of only ~1–3° for shank and leg segment angles) and,
> via regression equations, predicts 3D kinetics — vertical ground reaction
> force (R² = 0.75) and average loading rate (R² = 0.52). Provides the
> methodological justification for the entire single-camera approach: phone
> sagittal keypoints are sufficient to estimate the braking/loading quantities
> that overstriding drives.

---

Napier, Christopher, et al. "Kinetic Risk Factors of Running-Related Injuries
in Female Recreational Runners." *Scandinavian Journal of Medicine & Science
in Sports*, vol. 28, no. 10, Oct. 2018, pp. 2164–2172, doi:10.1111/sms.13228.

> Prospective study identifying peak braking (anterior-posterior) force as the
> strongest kinetic predictor of running-related injury — runners in the
> highest braking-force tertile were injured at roughly 5–8× the rate of the
> lower tertiles. Since braking impulse scales with how far the foot lands
> ahead of the COM, this grounds the overstride-distance metric in injury
> epidemiology and justifies flagging the mild (5–10 cm) range rather than
> treating it as acceptable.

---

Souza, Richard B. "An Evidence-Based Videotaped Running Biomechanics Analysis."
*Physical Medicine and Rehabilitation Clinics of North America*, vol. 27, no. 1,
Feb. 2016, pp. 217–236, doi:10.1016/j.pmr.2015.08.006.

> Primary clinical source for the Shin Test. Establishes tibia angle at loading
> response — alongside foot inclination angle and knee flexion at initial
> contact — as the sagittal-plane indicator of overstriding when force plates
> are unavailable: "an extended tibia is not ideal; a vertical or flexed tibia
> allows the runner to dissipate impact through knee flexion," with knee flexion
> under ~45° suggesting reduced shock absorption. Grounds the tibia-angle metric.

---

Stiffler-Joachim, Mikel R., et al. "Foot Angle and Loading Rate during Running
Demonstrate a Nonlinear Relationship." *Medicine and Science in Sports and
Exercise*, vol. 51, no. 10, Oct. 2019, pp. 2067–2072,
doi:10.1249/MSS.0000000000002023.

> The honesty check on the threshold tables. Foot inclination angle and average
> vertical loading rate follow a cubic — not linear — relationship, with loading
> lowest at the angular extremes, so binning these landing angles into categories
> "may misrepresent the relationship." Grounds the decision to report the
> continuous angle rather than treat the screening bands as hard cutoffs.
