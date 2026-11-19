# AeroTrack: UAV route reconstruction

Finds where each of 50 drone frames was taken on a satellite map and draws the drone's route.

## Setup

Needs Python 3.8+.

```bash
pip install -r requirements.txt
```

Download the input data ([Google Drive](https://drive.google.com/file/d/1h6xfYQfokDo5AzSv1Q-x66HluDpWSh87/view?usp=sharing)) and put it here.
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
3. **Position.** A homography found with MAGSAC (a robust RANSAC variant) gives how the frame sits on the map: position, rotation and scale. The frame's centre is the drone position and its 4 corners are the footprint.
4. **Clean-up.** Frames with too few matches are skipped, and so are mirrored or folded footprints. A frame whose footprint is much smaller or bigger than the typical one, or whose point is further from the route than the drone could fly (judged by the typical step between frames nearby), is dropped as an outlier. Points keep their real frame numbers (`crop_25.png` is point 25), and the route is dashed where frames are missing, so a gap stays visible.
5. **Smoothing.** Frame positions are drawn as found, since they are already accurate to about a pixel. Video input gets a 5-point moving average instead; START and FINISH are never moved.
6. **Drawing.** The route, points, START / FINISH and footprints are drawn on the map. The video moves a zoomed window along the interpolated route.

Very large maps are downscaled to at most 4000 px on the longest side for speed.
