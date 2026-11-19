# AeroTrack: UAV route reconstruction

Finds where each of 50 drone frames was taken on a satellite map and draws the drone's route.
Task details: [docs/TASK.md](docs/TASK.md).

## Setup

```bash
pip install -r requirements.txt
```

Download the input data (link in [docs/TASK.md](docs/TASK.md)) and put it here.
`data/` is git-ignored, so the images are never committed:

```
data/
  map.png        # satellite map
  frames/        # crop_0.png ... crop_49.png
```

## Run

```bash
python main.py --map data/map.png --frames data/frames
```

Results go to `output/` (also git-ignored):

- `output/route.png`: map with the route, numbered points, START / FINISH, frame footprints and a legend
- `output/route.mp4`: zoomed map window following the drone (pass `--video ""` to skip it)

## How it works

1. **Features.** SIFT finds distinctive points on the map and on each frame.
2. **Matching.** Each frame's points are matched to the map's points, keeping only clear matches (Lowe's ratio test, 0.75).
3. **Position.** A homography with RANSAC gives how the frame sits on the map: position, rotation and scale. The frame's centre is the drone position and its 4 corners are the footprint.
4. **Clean-up.** Frames with too few matches are skipped. A point that jumps far from both neighbours is dropped as an outlier. Points keep their real frame numbers, so a gap stays visible.
5. **Smoothing.** A 3-point moving average removes small jitter.
6. **Drawing.** The route, points, START / FINISH and footprints are drawn on the map. The video moves a zoomed window along the interpolated route.

Very large maps are downscaled to at most 4000 px on the longest side for speed.
