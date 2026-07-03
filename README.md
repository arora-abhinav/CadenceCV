# StrideLens

A research-grounded running form analysis tool. StrideLens extracts pose keypoints from a single side-view running video, computes biomechanical metrics, and reports results in the context of published research and consumer running-dynamics benchmarks.

**This is not a medical device, an injury predictor, or a substitute for assessment by a coach or physical therapist.** It measures and contextualizes objective biomechanical metrics — it does not diagnose.

## Overview

Given a side-view treadmill video, StrideLens reports:

- **Cadence** (steps per minute)
- **Vertical oscillation** (cm of vertical "bounce" per stride)
- **Foot strike pattern** (heel / midfoot+forefoot)
- **Ground contact detection** (contact vs. non-contact phase per gait cycle)
- **Overstriding** (unsupervised detection via tibial angle and heel-hip offset at contact)

Each metric is contextualized against published reference ranges and methodology, with citations.

## Pipeline

1. **Pose extraction** — YOLOv8x-pose (COCO-18, fine-tuned on COCO-WholeBody to add big toe keypoint) extracts 18 body keypoints per frame at `imgsz=1280`
2. **Signal cleaning** — Savitzky-Golay filtering (`window_length=11, polyorder=2`) smooths noisy keypoint trajectories
3. **Gait cycle detection** — `scipy.signal.find_peaks` (`distance=14`) on the smoothed ankle trajectory identifies footstrikes
4. **Cadence & vertical oscillation** — computed directly from gait cycle timing and hip trajectory amplitude; body-relative normalization (pixel distances divided by body-length-in-pixels) for scale invariance across runners and camera distances
5. **Per-strike feature extraction** (numpy) — at each strike frame: ankle-toe angle, knee flexion angle, tibial angle, ankle-hip horizontal offset, ankle vertical velocity, trunk angle, shin velocity
6. **Dataset construction** (pandas) — per-strike features aggregated into tables for classification and exploratory analysis. Raw unsmoothed keypoint arrays stored as `.npz` per video; labels stored separately in JSON
7. **Ground contact detection** — FootNet-inspired bidirectional LSTM (Bhosale et al., PLOS ONE) trained on 438 manually labeled gait cycles across 29 videos. Input: 4 signals per frame (ankle x/y velocity, tibial angle, shin velocity), resampled to fixed length of 40 timesteps with zero-padding and mask. Achieves 92% accuracy, contact F1=0.90 on held-out test set
8. **Strike pattern classification** (scikit-learn Random Forest) — binary classification (heel vs. midfoot+forefoot) using 8 per-strike features. Forefoot collapsed into midfoot class due to dataset imbalance (3.5% forefoot). Achieves 67% accuracy; primary limitation is absence of explicit toe keypoint for foot-ground angle computation — addressed by COCO-WholeBody fine-tuning (see Future Extensions)
9. **Overstriding detection** — unsupervised GMM clustering on two body-relative geometric features at the strike frame: tibial angle relative to vertical and normalized heel-hip horizontal offset. Two-cluster solution validated by domain expert inspection. Borderline cases (low cluster membership probability) flagged as uncertain rather than forced into a binary label
10. **Per-run aggregation** (pandas) — cadence, vertical oscillation, strike pattern distribution, contact time, overstriding rate
11. **Report generation** — JSON report with contextualized metrics + annotated video with skeleton overlay and per-stride classifications

## Models

### Ground contact detection (FootNet-inspired bidirectional LSTM)
Replicates FootNet (Bhosale et al., PLOS ONE 2021) using pose keypoints alone — no force plates required. Input: `(438, 40, 4)` tensor of gait cycles, zero-padded to fixed length with boolean mask. Bidirectional LSTM (hidden size=32, 1 layer) with per-timestep binary output. Trained with `BCEWithLogitsLoss` with `pos_weight` to handle 1:2 contact/non-contact imbalance. Velocities normalized by fps (pixels/second) for frame-rate agnosticism.

| Metric | Score |
|---|---|
| Accuracy | 0.92 |
| Non-contact F1 | 0.94 |
| Contact F1 | 0.90 |

### Strike pattern classifier (Random Forest)
Binary tabular classification (heel vs. midfoot+forefoot) at the strike frame. Video-level train/test split to prevent data leakage. Top features by importance: ankle-hip horizontal offset (0.19), knee flexion angle (0.18), ankle x velocity (0.13).

| Metric | Score |
|---|---|
| Accuracy | 0.67 |
| Heel F1 | 0.61 |
| Midfoot+Forefoot F1 | 0.71 |

**Known limitation**: absence of explicit toe keypoint prevents computation of the true foot-ground angle at contact, which is the most clinically meaningful discriminator between strike patterns. Accuracy ceiling is attributed to this limitation. Addressed in the dataset extension plan below.

### Overstriding detector (GMM, unsupervised)
Two-component Gaussian Mixture Model on tibial angle at contact and normalized heel-hip horizontal offset. Cluster assignment validated by domain expert. Borderline cases (membership probability below threshold) flagged as uncertain. No manual per-strike overstriding labels required.

## Dataset

- **29 videos** across self-filmed treadmill footage and external sources (YouTube Shorts, Instagram Reels via yt-dlp)
- **777 manually labeled strikes** stored in `strikefoot_data.json` with 8 biomechanical features, strike pattern, side (L/R), u-frame (toe-off), and keypoint confidence scores per entry
- **Train/test split**: video-level to prevent data leakage. Test set: Video9, instavid_13, instavid_15, Video3 (~148 strikes, ~20%)
- **Class distribution**: midfoot 69.4%, heel 27.2%, forefoot 3.5%

## Dataset Extension Plan

Fine-tuning YOLOv8x-pose on COCO-WholeBody to add an 18th keypoint (big toe) to the standard COCO-17 skeleton. This enables direct computation of the foot-ground angle at contact — the primary missing feature for strike pattern classification. COCO-WholeBody annotations filtered to retain COCO-17 keypoints plus left/right big toe and left/right heel from `foot_kpts`. Annotations with `foot_valid=False` are zeroed out. Images sourced directly from COCO train2017/val2017.

## Reference context

- Cadence: widely studied, but the commonly cited "180 spm" figure derives from observations of elite runners and is not a universal target — comparisons here are pace-aware where possible
- Vertical oscillation: typical range 6-13cm; lower values at a given pace are generally associated with better running economy
- Overstriding: associated with increased impact loading and braking forces in the literature; detected here via tibial inclination and heel-hip geometry at contact rather than force plate data
- FootNet: Bhosale et al. (2021), PLOS ONE — bidirectional LSTM for foot event detection, foot-strike bias 0ms, RMSE 5ms

## Tech stack

| Layer | Tool |
|---|---|
| Pose estimation | YOLOv8x-pose (COCO-18, fine-tuned) |
| Video I/O | OpenCV |
| Signal processing | scipy |
| Numerical features | numpy |
| Dataset / aggregation | pandas |
| Strike pattern classifier | scikit-learn (Random Forest) |
| Contact detection | PyTorch (bidirectional LSTM) |
| Overstriding detector | scikit-learn (GMM) |
| Experiment tracking | MLflow |
| Backend | FastAPI |
| Frontend | HTML/JS |

## Status

In development.

**Completed**:
- [x] Pose extraction pipeline (YOLOv8x-pose, COCO-17)
- [x] Signal cleaning and gait cycle detection
- [x] Cadence and vertical oscillation
- [x] Manual labeling of 777 strikes across 29 videos
- [x] Ground contact detection LSTM (92% accuracy, contact F1=0.90)
- [x] Strike pattern Random Forest (67% accuracy)
- [x] COCO-WholeBody dataset filtering and image download

**In progress**:
- [ ] YOLOv8x-pose fine-tuning on COCO-18
- [ ] Overstriding GMM clustering and expert validation
- [ ] Per-run aggregation and report generation
- [ ] FastAPI integration
- [ ] Frontend

## Future extensions

- Re-run strike pattern classifier with explicit foot-ground angle after COCO-18 fine-tuning
- Bilateral symmetry analysis (siamese LSTM on paired left/right gait cycles, front-view footage)
- Ground contact time computation
- Larger labeled dataset across multiple runners for improved generalization
