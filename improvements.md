This is genuinely achievable but requires thinking across three layers:
  accuracy, infrastructure, and product. Here's an honest breakdown:

  ---
  What's Actually Holding It Back Right Now

  Everything is in pixels, not real-world units

  Step length, vertical oscillation, and GCT in milliseconds are meaningless
  without a pixel-to-metre conversion. Run labs use calibrated treadmills. You
  need either:
  - The athlete's height as a reference (entered by the user) — use hip-to-ankle
  distance in pixels as your scale factor
  - A calibration marker at a known distance in frame
  - Known camera distance + focal length via EXIF

  Without this, your metrics are relative, not absolute, and can't be compared
  to clinical norms.

  Single 2D camera
  
  Run labs use multiple camera angles or motion capture. A side-view phone gives
  you sagittal plane only — you're blind to lateral drift, crossover gait, and
  hip drop (Trendelenburg). A second camera at 45° or front-facing would unlock
  an enormous amount of additional diagnostics.

  No normative database

  A run lab doesn't just give you numbers — it tells you where you sit relative
  to the population. You need a database of labelled athlete data with metrics
  segmented by speed, age, and experience level so you can say "your GCT is in
  the 30th percentile for recreational runners at this pace."

  ---
  Model Robustness (What Will Break in the Wild)
  
  - YOLO will fail on low-light videos, loose clothing, partial occlusion, and
  non-side-view angles. You need a video quality gate at ingestion — reject
  videos that aren't usable rather than silently producing bad output.
  - Your FootNet predictions need confidence scores, not just binary outputs.
  torch.sigmoid already gives you a probability — surface it. Flag frames where
  confidence is below a threshold rather than forcing a hard prediction.
  - Temporal smoothing — right now a single noisy frame can flip a prediction. A
  short median filter over the raw sigmoid outputs before thresholding would
  make predictions far more stable.
  - Handle multiple people in frame — currently you take keypoints[0], which is
  whoever YOLO puts first. That's not guaranteed to be the runner you want.

  ---
  Infrastructure for Scalability
  
  Async video processing

  Video processing is slow (minutes per video). You need a job queue — Celery +
  Redis is the standard. User uploads video → job queued → processed in
  background → results returned when ready. A synchronous HTTP endpoint will
  time out.

  Storage

  Videos go to S3 or equivalent object storage, never local disk. Results go to
  PostgreSQL. Models load once at worker startup, not per request.

  API design

  FastAPI is the natural fit given your Python stack. Three core endpoints:
  - POST /analyse — accepts video, returns job ID
  - GET /results/{job_id} — returns structured JSON of all metrics
  - GET /history/{athlete_id} — longitudinal tracking across sessions

  Containerisation

  Docker with your model weights baked in. GPU inference via CUDA if you want
  sub-minute turnaround on a T4 instance.

  ---
  Output That Actually Competes With Labs
  
  Run labs produce PDF reports with annotated video, metric charts, and
  recommendations. You need:

  - Annotated video output — skeleton overlay with contact/flight phase
  coloured, strike type labelled per step, overstriding flagged in red. OpenCV
  can do this frame by frame.
  - Per-session metric summary — cadence, GCT mean/SD, contact ratio, strike
  distribution (% heel/mid/fore), symmetry score, vertical oscillation, step
  length
  - Longitudinal dashboard — plot the same metrics across sessions so athletes
  can see improvement. This is something run labs genuinely can't do affordably
  because a lab visit costs $300.
  - Percentile rankings — requires your normative database but is the single
  most interpretable output for a non-expert athlete

  ---
  Genuine Competitive Advantages Over Run Labs
  
  ┌──────────────────────────────────┬─────────────────────────────────┐
  │             Run Lab              │           Your System           │
  ├──────────────────────────────────┼─────────────────────────────────┤
  │ $200–500 per session             │ Near zero marginal cost         │
  ├──────────────────────────────────┼─────────────────────────────────┤
  │ Controlled treadmill environment │ Any flat surface, any phone     │
  ├──────────────────────────────────┼─────────────────────────────────┤
  │ One-off snapshot                 │ Unlimited longitudinal tracking │
  ├──────────────────────────────────┼─────────────────────────────────┤
  │ Results days later               │ Minutes                         │
  ├──────────────────────────────────┼─────────────────────────────────┤
  │ Clinician required               │ Automated                       │
  └──────────────────────────────────┴─────────────────────────────────┘
  
  The longitudinal angle is your biggest differentiator. A run lab gives you a
  snapshot. You can give an athlete a session-by-session history of every
  metric, tracked across months of training. No run lab can match that at scale.

  ---
  What You Genuinely Cannot Replace Without Hardware
  
  Be honest about this in any product framing:
  - Ground reaction forces — require force plates. You cannot compute true
  braking impulse or loading rate from video alone.
  - 3D joint angles — a single side-view camera gives you sagittal plane only.
  True 3D requires stereo cameras or IMUs.
  - Metabolic cost — requires a metabolic cart. You can correlate GCT and
  contact ratio as proxies for economy but can't measure it directly. 

  The honest positioning is: everything a run lab measures kinematically from 
  video, at 1% of the cost, with longitudinal tracking they can't offer.

  ---
  Learning Roadmap in Your Stack
  
  1. Pixel-to-metre calibration — numpy geometry, immediate impact
  2. Confidence scoring + temporal smoothing — torch sigmoid outputs, median
  filter
  3. Annotated video export — OpenCV, makes output tangible
  4. FastAPI + Celery backend — the infrastructure leap
  5. Normative database + percentile ranking — pandas, requires data collection
  6. Longitudinal dashboard — product layer, could be a simple Streamlit app
  initially
  7. Second camera angle — the biggest accuracy unlock, requires homography and
  multi-view geometry (numpy-heavy, very learnable)

✻ Cogitated for 55s
