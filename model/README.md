# Installed model configuration

Your weights are recognized directly under their original filenames:

- best_players_gk_1280_s_e300.pt (players)
- ball_tracking_1280_e300.pt (ball)
- pitch_kpts32_y8s_640_e500_AO.pt (pitch)

No rename is necessary. Short filenames below take precedence if both exist.
Nothing is downloaded automatically.

The included pitch_points.json preserves the output order of the original
SoccerPitchConfiguration.vertices, converted from centimetres to metres:
https://github.com/Simo-03/football-player-detection/blob/main/src/core/pitch.py

It uses the source's **120 x 70 m** template, including its 20.15 m penalty-box
depth and 41 m penalty-box width. These reproduce the original configuration;
they are not measured dimensions of your video. For accurate physical distances,
calibrate the template to the actual field. Changing dimensions alone without
updating landmark coordinates is not sufficient.

The JSON order follows model output indices, not sorted display labels.
For different trained models, the following contract applies:

- players.pt: Ultralytics detection model. Class names should include player,
  goalkeeper, referee (ball may also be present).
- ball.pt: optional dedicated ball detector, with a class named ball.
- pitch.pt: optional Ultralytics pose model for pitch landmarks.
- pitch_points.json: required when pitch.pt is used. Array of [x_metres, y_metres]
  pairs in EXACTLY the same order as your pose dataset's keypoints. Coordinates
  start at the top-left of the minimap; x increases along pitch length and y
  across pitch width. Use null for any keypoint you want to exclude.

Do not copy an arbitrary 32-point layout: the numbering comes from your dataset.
Any number of landmarks is supported, with at least four usable correspondences.
Set matching pitch dimensions in the UI/CLI (default 120 by 70 metres).
Pitch mapping stays disabled until you provide this mapping. Detection and
tracking work with players.pt alone. No training or weight downloads are included.
