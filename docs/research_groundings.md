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

### Works Cited

Barrett, Justin, et al. "Implementation of 2D Running Gait Analysis in
Orthopedic Physical Therapy Clinics." *International Journal of Sports
Physical Therapy*, vol. 18, no. 3, 1 June 2023, pp. 582–595,
doi:10.26603/001c.74726.

> Establishes the validity of 2D video analysis for sagittal plane variables
> including tibial inclination and knee flexion at initial contact. Directly
> supports the use of a single side-view phone camera as the measurement
> instrument and validates that 2D keypoint-derived metrics map reliably to
> clinical overstriding risk.

---

Edwards, Brent, et al. "Effects of Stride Length and Running Speed on Tibial
Stress Fracture Probability." *Medicine and Science in Sports and Exercise*,
vol. 41, no. 12, Dec. 2009, pp. 2177–2184,
doi:10.1249/MSS.0b013e3181abbeb0.

> Foundational study linking extended horizontal foot-to-COM distance to steep
> loading rate spikes and statistically elevated probability of tibial stress
> fractures. Primary source for the severe overstride threshold (> 10 cm) and
> the clinical framing of overstriding as a bone stress injury risk factor.

---

Heiderscheit, Bryan C., et al. "Effects of Step Rate Manipulation on Joint
Mechanics during Running." *Medicine and Science in Sports and Exercise*,
vol. 43, no. 2, Feb. 2011, pp. 297–302,
doi:10.1249/MSS.0b013e3181ebedf5.

> Clinical trial demonstrating that increasing cadence by 5–10% significantly
> reduces the horizontal distance between the foot and the COM at initial
> contact, thereby reducing joint loading at the hip and knee. Grounds cadence
> as the primary exercise intervention recommended when overstriding is flagged
> — increasing step rate is the most evidence-backed cue for reducing
> overstride distance without requiring the runner to consciously change their
> foot placement.

---

Napier, Christopher, et al. "Kinematic Predictors of Frontal and Sagittal
Plane Pathomechanics in Running." *Journal of Orthopaedic & Sports Physical
Therapy*, vol. 48, no. 8, Aug. 2018, pp. 612–621,
doi:10.2519/jospt.2018.8004.

> Proves that elevated horizontal foot-to-COM distance at initial contact is a
> direct kinematic predictor of peak anterior-posterior braking forces and
> overall running injury risk. Grounds the mild overstride threshold (5–10 cm)
> in injury epidemiology and justifies flagging this range rather than treating
> it as acceptable.

---

Wille, Corrine M., et al. "Ability of a 2D Video-Based Gait Analysis to
Predict 3D Joint Angles and Forces during Running." *Journal of Athletic
Training*, vol. 49, no. 5, Oct. 2014, pp. 611–617,
doi:10.4085/1062-6050-49.3.34.

> Validates the use of a single 2D camera to accurately approximate complex 3D
> internal joint forces, specifically isolating the braking moments produced
> by overstriding. Provides the methodological justification for the entire
> 2D video analysis approach used in this pipeline — that sagittal plane
> keypoint data from a phone camera is sufficient to detect clinically
> meaningful overstriding.
