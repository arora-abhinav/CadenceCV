# StrideLens

A research-grounded running form analysis tool. StrideLens extracts pose keypoints from a single side-view running video, computes biomechanical metrics, and reports results in the context of published research and consumer running-dynamics benchmarks.

**This is not a medical device, an injury predictor, or a substitute for assessment by a coach or physical therapist.** It measures and contextualizes objective biomechanical metrics — it does not diagnose.

## Overview

Given a side-view video and the runner's height (for calibration), StrideLens reports:

- **Cadence** (steps per minute)
- **Vertical oscillation** (cm of vertical "bounce" per stride)
- **Foot strike pattern** (heel / midfoot / forefoot)
- **Overstriding** (whether the foot lands ahead of the body's center of mass)

Each metric is contextualized against published reference ranges and methodology, with citations.

## Pipeline

1. **Pose extraction** — MediaPipe Pose extracts 33 body keypoints per frame (downsampled to ~15fps)
2. **Signal cleaning** — Savitzky-Golay filtering smooths noisy keypoint trajectories
3. **Gait cycle detection** — `scipy.signal.find_peaks` on the smoothed ankle trajectory identifies footstrikes
4. **Cadence & vertical oscillation** — computed directly from gait cycle timing and hip trajectory amplitude
5. **Per-strike feature extraction** (numpy) — at each strike frame: ankle-toe angle, knee flexion angle, tibial angle, ankle-hip horizontal offset, ankle vertical velocity, trunk angle
6. **Dataset construction** (pandas) — per-strike features aggregated into tables for classification and exploratory analysis
7. **Strike pattern classification** (scikit-learn) — logistic regression and random forest, 3-class (heel/mid/forefoot)
8. **Overstriding classification** (PyTorch LSTM + sklearn baseline) — sequence model over a 12-frame window around each strike, predicting overstride vs. neutral
9. **Per-run aggregation** (pandas) — cadence, vertical oscillation, strike pattern distribution, overstriding rate
10. **Report generation** — JSON report with contextualized metrics + annotated video with skeleton overlay and per-stride classifications

## Classifiers

### Strike pattern (heel / mid / forefoot)
Tabular classification using ankle-toe and ankle-knee angles at the strike frame. Labels assigned manually from annotated video. Compares logistic regression vs. random forest.

### Overstriding (overstride / neutral)
Sequence classification over a 12-frame window (8 frames before strike, 4 after, resampled to fixed length) using 6 features per frame: knee flexion angle, tibial angle, foot inclination angle, ankle-hip horizontal offset, ankle vertical velocity, trunk angle.

Labels follow a published video-assessment method: a vertical line drawn upward from the ankle (lateral malleolus) at the strike frame — if it falls within the pelvis, the landing is neutral; if anterior to the pelvis, it indicates overstriding.

**Known limitation**: this labeling method does not account for trunk flexion (which shifts true center of mass) and may be less reliable for midfoot/forefoot strikers. This is a limitation of the published methodology itself, documented here for transparency.

A small PyTorch LSTM (32 hidden units, 1 layer) is trained on the windowed sequences and compared against a flattened-feature sklearn baseline (logistic regression / random forest on the same window, flattened to a single vector).

## Reference context

- Cadence: widely studied, but the commonly cited "180 spm" figure derives from observations of elite runners and is not a universal target — comparisons here are pace-aware where possible
- Vertical oscillation: typical range 6-13cm; lower values at a given pace are generally associated with better running economy
- Overstriding: defined as the horizontal distance the foot lands ahead of the body's center of mass; associated with increased impact loading and braking forces in the literature

## Tech stack

| Layer | Tool |
|---|---|
| Pose estimation | MediaPipe Pose |
| Video I/O | OpenCV |
| Signal processing | scipy |
| Numerical features | numpy |
| Dataset / aggregation | pandas |
| Strike pattern classifier | scikit-learn |
| Overstriding classifier | PyTorch (LSTM) + scikit-learn baseline |
| Backend | FastAPI |
| Frontend | HTML/JS |

## Calibration

Vertical oscillation (pixels -> cm) is calibrated using the runner's self-reported height as a scale reference.

## Status

In development.

**Roadmap**:
- [ ] Pose extraction pipeline
- [ ] Signal cleaning, gait cycle detection, cadence & vertical oscillation
- [ ] Strike pattern feature extraction + classifier
- [ ] Overstriding window extraction + labeling
- [ ] Overstriding LSTM + sklearn baseline comparison
- [ ] Per-run aggregation & report generation
- [ ] FastAPI integration
- [ ] Frontend

## Future extensions

- Ground contact time, additional joint angles
- Multi-angle video support
- Larger labeled dataset across multiple runners for improved generalization
