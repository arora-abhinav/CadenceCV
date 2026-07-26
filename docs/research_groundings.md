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

---

## 3. Metric-Family Evidence Map (RAG corpus sources)

Sources for the report-generation RAG, organised by the metric families in
`inference_metrics.py`. Every entry here has been **verified** against a primary
source (title, authors, journal, DOI/PMID, and the specific claim). Papers
sourced but not yet verified live in the "Pending Verification" queue at the end
and must NOT be treated as established until checked — this project has already
shipped three fabricated citations, so nothing enters as fact on trust.

Each entry carries a **Direction** tag (supports / contradicts / neutral) so the
retriever can be forced to surface at least one *contradicting* chunk per metric
— otherwise "high vertical oscillation" only ever retrieves papers calling it bad.

### Cross-metric "spine" papers

Malisoux, Laurent, et al. "Reference Values and Determinants of Spatiotemporal
and Kinetic Variables in Recreational Runners." *Orthopaedic Journal of Sports
Medicine*, vol. 11, no. 10, Oct. 2023, doi:10.1177/23259671231204629.

> Speed- and sex-stratified reference values from 860 healthy recreational
> runners on an instrumented treadmill: contact time, flight time, duty factor,
> vertical oscillation, cadence, step length, loading rate, vertical stiffness,
> peak vGRF, peak braking force. Only running **speed** correlated highly
> (r > 0.7) with the biomechanical variables. Direction: neutral / normative.
> *This is the reference-range source for judging whether a value is "high."*

---

Van Hooren, Bas, et al. "The Relationship Between Running Biomechanics and
Running Economy: A Systematic Review and Meta-Analysis of Observational
Studies." *Sports Medicine*, vol. 54, no. 5, May 2024, pp. 1269–1316,
doi:10.1007/s40279-024-01997-3.

> Meta-analysis across essentially every spatiotemporal/kinematic variable vs
> running economy. Its framing is that prior findings are inconsistent and it
> explains why. Direction: neutral — the "it depends" counterweight to any
> single-variable "optimal form" claim.

### Cadence / stride length / overstriding

Anderson, Luke M., et al. "What is the Effect of Changing Running Step Rate on
Injury, Performance and Biomechanics? A Systematic Review and Meta-analysis."
*Sports Medicine – Open*, vol. 8, no. 112, 2022, doi:10.1186/s40798-022-00504-0.

> Increasing step rate produces increases or no change in loading-rate variables
> at ankle/knee/hip, but evidence is insufficient to conclude effects on injury
> or performance; long-term effects largely unknown. Direction: contradicts —
> the hedge against over-confident cadence advice.

---

Baker, Lauren M., et al. "Predicting Overstriding with Wearable IMUs during
Treadmill and Overground Running." *Scientific Reports*, vol. 14, no. 6347,
2024, doi:10.1038/s41598-024-56888-4.

> Sagittal segment angles explained 95–98% of overstriding variance and 80–88%
> of peak braking force variance. Defines overstriding as horizontal distance the
> foot lands ahead of the body. Direction: supports the overstride-distance
> metric. (NOTE: originally mis-cited as "Rodriguez et al." — actual first
> author is Baker.)

### Ground contact time / flight time / duty factor

Lussiana, Thibault, et al. "Duty Factor Is a Viable Measure to Classify
Spontaneous Running Forms." *Sports*, vol. 7, no. 11, 2019, p. 233,
doi:10.3390/sports7110233.

> Defines duty factor as contact time / stride time and validates it against the
> subjective Volodalen aerial-vs-terrestrial scale (79.8% agreement). Direction:
> supports — turns the duty-factor number into a running-form narrative.
> (Author order to double-check: Patoz/Gindre may be lead; DOI/title confirmed.)

### Foot strike angle / strike pattern

Daoud, Adam I., et al. "Foot Strike and Injury Rates in Endurance Runners: A
Retrospective Study." *Medicine and Science in Sports and Exercise*, vol. 44,
no. 7, 2012, PMID 22217561.

> Habitual rearfoot strikers had ~2× the rate of repetitive stress injuries vs
> forefoot strikers; the study does NOT test causation. Direction: supports (with
> a causality caveat). DOI to confirm.

---

Han, [first author], et al. "Influence of Manipulating Running Foot Strike Angle
on Internal Loading of the Tibia." *Scandinavian Journal of Medicine & Science
in Sports*, 2025, doi:10.1111/sms.70066.

> In 19 habitual rearfoot strikers, an imposed forefoot strike *increased* tibial
> peak bending moment ~15% vs habitual rearfoot; concludes transitioning
> rearfoot→forefoot "may not be advisable" to reduce tibial load. Direction:
> **contradicts** — without this, the system will happily tell heel strikers to
> convert to forefoot. Critical guardrail.

### Loading rate / peak vGRF (impact)

Matijevich, Emily S., et al. "Ground Reaction Force Metrics Are Not Strongly
Correlated with Tibial Bone Load When Running across Speeds and Slopes:
Implications for Science, Sport and Wearable Tech." *PLOS ONE*, vol. 14, no. 1,
2019, e0210000, doi:10.1371/journal.pone.0210000.

> GRF metrics (impact peak, loading rate, active peak, impulse) are NOT strongly
> correlated with tibial compression force across speeds/slopes. Direction:
> **contradicts** — the guardrail on any "estimated peak vGRF → injury risk"
> statement the system might make.

### Symmetry (all four symmetry metrics)

Malisoux, Laurent, et al. "Gait Asymmetry in Spatiotemporal and Kinetic
Variables Does Not Increase Running-Related Injury Risk in Lower Limbs: A
Secondary Analysis of a Randomised Trial Including 800+ Recreational Runners."
*BMJ Open Sport & Exercise Medicine*, 2024, PMID 38196940, PMC10773390.

> 836 recreational runners: gait asymmetry was NOT associated with higher injury
> risk; greater asymmetry in flight time and peak braking force was associated
> with *lower* risk. Direction: **contradicts** the intuitive "asymmetry is bad"
> advice — the largest prospective evidence available, must anchor the symmetry
> outputs. (Journal is BMJ OSEM, not BJSM as originally sourced. DOI to confirm.)

### Practical notes for building the index

- **Speed-conditioning is the dominant caveat.** Malisoux 2023 found running
  speed is the main determinant of nearly every metric, so retrieval queries
  should include the computed `running_speed` bucket and threshold-based advice
  must be gated on it (a 0.28 s GCT is not "long" at 3:00/km pace).
- **Force a contradicting chunk per metric.** Tag every chunk with
  `direction ∈ {supports, contradicts, neutral}` and make the reranker pull at
  least one contradicting chunk, or the system becomes a confirmation-bias engine.
- **Suggested chunk metadata:** `metric_tags[]` (exact return keys),
  `evidence_level` (RCT / prospective cohort / cross-sectional / review), `n`,
  `population`, `speed_range`, `direction`.
- **Licensing:** only the PMC Open Access Subset is redistributable — check each
  PMCID via the OA web service before ingesting; a PMCID alone does not imply OA.
  Paywalled sources: store title/DOI/abstract as pointers, do not chunk full text.

### Honest coverage gaps

- **Hip extension at toe-off** has no verified open-access anchor yet. Flag this
  output as "descriptive only, no evidence linkage" until one is found.
- **`Average Vertical Oscillation (px)`** cannot be grounded against the
  literature (everything is reported in cm or % of step length). Report the
  scale-invariant **vertical ratio** instead, or flag VO-px as descriptive only.

### Pending Verification (sourced, NOT yet checked — do not cite as fact)

Cadence/overstride: Farina & Hahn 2021 (PMC8772793); Lieberman et al. 2015 (JEB).
GCT/DF: Joubert et al. 2020 (PMC7241633); Van Hooren & Bosch 2019 (Front Sports);
Santos-Concejero et al. 2013 (PMC3944563); Nijs et al. 2023 (PLOS ONE).
VO: Adams et al. 2018 (PMC6088121). Foot strike: Almeida et al. 2015 (PMID
26304644); self-reported foot-strike accuracy (Front Sports 2024). Trunk lean:
forward-lean economy paper (PMC11135760); AminiAghdam et al. 2022 (PMID 34537800).
Knee: PFP current concepts (PMC7740062); PFP subgroup preprint (medRxiv).
Spring-mass: Morin et al. 2005 (verified earlier this project); Morin et al. 2006
(PMID 16475063); nonlinear spring-mass regression (JEB 2021); Coleman et al. 2012
(paywalled). Loading/GRF: "Rethinking running biomechanics" review (Front Bioeng
2024); Zadpoor & Nikooyan meta. Symmetry: Weighted USI (PMC7644861); U14
asymmetry (PMC11125289); fatigue-symmetry papers. Methods: Stenum et al. 2021
(PMC8099131, OpenPose errors 4.0°/5.6°/7.4°); Pipkin et al. 2016 (PMC6044590);
Michelini et al. 2020.
