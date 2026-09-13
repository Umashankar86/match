# Touchline — football video analysis

A fresh Python implementation with a Streamlit interface and CLI. Supply your own
trained models after training. This project includes **no model weights, training
jobs, dataset downloads, or automatic model downloads**.

Your three supplied weights are now connected under their original filenames.
The matching `model/pitch_points.json` is included. No renaming is necessary.

## Start on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit. Upload a video or enter a local path,
choose CPU/NVIDIA GPU/Apple MPS, and click **Analyze video**. Start with the default
750-frame preview. Set maximum frames to 0 for the entire video.

For NVIDIA acceleration, install the CUDA-compatible PyTorch build for your
machine using https://pytorch.org/get-started/locally/ and select device `0`.

## Add models when ready

```text
model/
  players.pt          required: player / goalkeeper / referee detector
  ball.pt             optional: dedicated ball detector
  pitch.pt            optional: field landmark pose detector
  pitch_points.json   required only when pitch.pt is present
```

For new training runs, rename each corresponding `best.pt` to the filename above.
The existing `best_players_gk_1280_s_e300.pt`, `ball_tracking_1280_e300.pt`, and
`pitch_kpts32_y8s_640_e500_AO.pt` filenames are also recognized automatically.
Models must be Ultralytics-compatible. Class IDs are read from the model's class
names; supported aliases live in `football/pipeline.py:role`.
If your player detector also includes ball, it supplies ball detections when
the dedicated ball model is absent.

See [model instructions](model/README.md). The supplied model's 32-landmark mapping
is configured already, with **120 x 70 m** template defaults matching its original
project. This is a reference template, not a measurement of your match's field.
For a different model, pitch mapping requires real-world
coordinates in your own dataset's keypoint order, regardless of point count.
This is a small configuration file, not an image annotation job. Extract the
keypoint definitions/order from your chosen dataset and map those landmarks to
pitch metres. Do not guess the ordering. Set pitch dimensions to match the file.

## What it does

- Player/referee/goalkeeper detection with BoT-SORT IDs and camera-motion compensation.
- Optional ball detection with overlapping 640-pixel tiles.
- Automatic two-team jersey colour clustering after a brief warm-up.
- Per-frame pitch homography with confidence, degeneracy and inlier checks.
- Tactical minimap, conservative observed player distance and speed.
- Proximity-based possession, pass and turnover estimates.
- Annotated video, CSV reports, per-player heatmaps/trajectories, pass networks and ZIP download.

Goalkeepers remain team-unknown because their kits differ. No appearance ReID
model is loaded. Team IDs are arbitrary colour clusters, not named clubs.
Pitch mapping is unavailable on frames with insufficient landmarks; old camera
transforms are not reused. Heatmaps are only generated for mapped player tracks.

## CLI

```powershell
.\.venv\Scripts\python.exe main.py --video "C:\videos\match.mp4" --device cpu --max-frames 750
```

Use `--device 0` for NVIDIA, `--model-dir PATH` for alternate weights, and
`--no-ball-tiles` for faster full-frame ball inference. Run `--help` for all flags.
Each run gets a unique folder under `outputs/`; previous results are preserved.

```text
outputs/<run>/
  analysis.mp4     annotated source + tactical panel; no audio
  tracks.csv       frame positions, role, team, confidence, pitch metres, speed
  events.csv       estimated passes and turnovers
  players.csv      observed valid movement distance per track
  summary.json     run summary and limitations
  heatmap_<id>.png optional pitch occupancy heatmaps
  trajectory_<id>.png observed pitch movement paths
  pass_network_team_<team>.png estimated passes between mean track positions
```

CSV team IDs are 0 and 1 (UI labels Team 1 and Team 2). Empty fields mean unknown,
not zero. Possession percentages exclude unassigned/unknown periods. Events are
heuristics, not verified match statistics. Detection accuracy, identity switches,
camera cuts and calibration noise can affect all analytics. Tracks are not
guaranteed to identify the same person throughout a match. Ball speed and full
3D ball motion are not reported; airborne balls do not lie on the pitch plane.

MP4 export uses OpenCV MPEG-4 encoding; play with VLC if browser playback is
unsupported. Files omit source audio. Uploads are temporary and removed after
processing; original local input videos are not changed. Interrupted/failed runs
may leave partial outputs; FAILED.txt identifies processing errors.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests use synthetic geometry/video and a fake detector. They never download or
load weights. Real model accuracy must be evaluated after you supply your models.

To check all three installed models with synthetic video, run:

```powershell
.\.venv\Scripts\python.exe verify_models.py
```

This runs actual local weights through tracking, ball and pitch inference,
then checks exported video frames and reports. It does not test football accuracy.

Implementation references: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/),
[prediction](https://docs.ultralytics.com/modes/predict/), and
[Streamlit](https://docs.streamlit.io/). No source code from the linked football
repository is required.
