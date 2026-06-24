# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**StrideLens** — a running biomechanics analysis pipeline that extracts pose keypoints from side-view running videos and computes cadence, vertical oscillation, foot-strike pattern, and overstriding. The project is in active data collection/labelling phase; classifiers are not yet trained end-to-end.

## Environment

The virtual environment is `.cadence_cv/` (not `venv/` or `env/`). Activate with:
```bash
source .cadence_cv/bin/activate
```

Run Jupyter notebooks:
```bash
jupyter notebook
```

Key packages: `ultralytics` (YOLOv8 pose), `opencv-python`, `scipy`, `numpy`, `pandas`, `torch`, `scikit-learn`, `mediapipe`.

## Architecture

### Data flow

```
Videos/ (MP4s, gitignored)
  → video_extracter.py        # YOLO pose → 8 keypoint arrays + metadata
  → data_extraction_1_2_3.ipynb  # FootNet LSTM dataset + manual gait-cycle labelling
  → strikefoot_type_trainer.ipynb # Strike-type features + cadence/VO metrics
  → Json Data/footnet_data.json   # Per-frame metrics keyed by video + frame index
  → Json Data/strikefoot_data.json # Per-strike classifications (h/m/f) + biomechanical metrics
```

### Key files

| File | Role |
|---|---|
| `Scripts/video_extracter.py` | Loads `yolo26n-pose.pt`, streams a video, returns 8 keypoint arrays (left/right ankle, hip, knee, shoulder) as numpy arrays plus duration and frame count |
| `Scripts/numpy_encoder.py` | `NumpyEncoder` — custom `json.JSONEncoder` subclass required when serialising any numpy array/scalar to JSON |
| `Scripts/data_extraction_1_2_3.ipynb` | FootNet-style dataset builder: extracts per-frame metrics (ankle velocities, tibial angles, shin velocity), then uses an interactive OpenCV window for manual gait-cycle labelling (key `u`/`d`/`c`), writes to `footnet_data.json` |
| `Scripts/strikefoot_type_trainer.ipynb` | Cadence/VO pipeline (Savitzky-Golay + `find_peaks` on ankle Y trajectory), per-strike feature extraction (knee flexion, tibial angle, ankle-hip offset, trunk angle, pre-strike velocity/approach angle), manual strike-type labelling (key `h`/`m`/`f`/`s`/`q`), writes to `strikefoot_data.json` |
| `yolo26n-pose.pt` | YOLOv8-nano pose model (also duplicated in `Scripts/`); loaded by path string — keep it in the repo root |

### YOLO keypoint indices (YOLOv8 convention)

| Index | Joint |
|---|---|
| 5 | left shoulder |
| 6 | right shoulder |
| 11 | left hip |
| 12 | right hip |
| 13 | left knee |
| 14 | right knee |
| 15 | left ankle |
| 16 | right ankle |

### JSON data schemas

**`footnet_data.json`** — keyed `Video1` … `VideoN`, each frame index maps to a list of `(metric_name, value)` tuples followed by a side label `"L"` or `"R"`.

**`strikefoot_data.json`** — keyed by video string (e.g. `"Video 1"`), each entry contains all metric arrays from `results` plus a `"Strikefoot Classifications"` dict mapping frame index → `"h"`, `"m"`, `"f"`, or `"s"`.

### Manual labelling conventions

**Gait cycle labelling** (`data_extraction_1_2_3.ipynb`):
- `u` — foot lifts off (toe-off)
- `d` — foot contacts ground (strikefoot) → completes a cycle
- `c` — finish labelling this video

**Strike type labelling** (`strikefoot_type_trainer.ipynb`):
- `h` — heel strike
- `m` — midfoot strike
- `f` — forefoot strike
- `s` — skip (ambiguous)
- `q` — quit and save progress

### Classifiers planned

1. **FootNet-style LSTM** — identifies strikefoot events from per-frame metrics (4 features: ankle velocities ×2, shin velocity, tibial angle); bidirectional, 2-layer, 32 hidden units
2. **Strike-type classifier** — random forest / logistic regression on per-strike tabular features
3. **Overstriding classifier** — LSTM over a 12-frame window centred on each strike

## Important implementation notes

- `extract_keypoints` returns `"Source"` (a YOLO generator) in its dict — this generator is exhausted after iteration and cannot be re-used.
- Video paths are hardcoded absolute paths in the notebooks; update them when running on a different machine.
- The `Videos/` directory is gitignored. The model weights `yolo26n-pose.pt` are committed (7.5 MB).
- Always use `NumpyEncoder` when calling `json.dump` with any output from the pipeline.
- Cadence is computed as total ankle peaks (both feet) divided by video duration in seconds — result is steps per second, not steps per minute; multiply by 60 for SPM.
