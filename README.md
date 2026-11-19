# AeroTrack: UAV route reconstruction

Finds where each of 50 drone frames was taken on a satellite map and draws the drone's route.
It also works on a video of the map filmed from a screen (advanced level, see below).
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
- `output/route.csv`: position of every frame on the map, in map pixels (empty if not found)

## Advanced level: track a video

The idea: show the map full screen, film it with a phone while moving the phone like a drone
flying over the city, then let the program find every moment of the video on the map.

### 1. Record the video (step by step)

1. Open `data/map.png` on a computer and make it **full screen** (Windows Photos: press `F11`).
   Turn screen brightness up. Close or hide anything that covers the map.
2. Turn off the room light behind you or close the curtains if you see reflections of a lamp
   or window on the screen. A little glare is fine, a big white spot is not.
3. On the phone, use the normal **video** mode (not slow motion, no filters, no zoom).
   Landscape or portrait both work. 1080p and 30 fps are plenty.
4. Hold the phone **10 to 30 cm from the screen**, looking straight at it. The camera should see
   only part of the map, about a quarter to a half of the screen, like a drone looking down.
5. Tap the screen of the phone to **focus** on the map before you start moving.
6. Press record and move the phone **slowly** along a path over the map, like a drone flight.
   Count about 3 to 5 seconds to cross a quarter of the screen. You can turn the phone too.
   Avoid fast jerks: blurry frames are skipped, but if all of them are blurry nothing is found.
7. Keep the whole flight on the map. If the camera only sees the black border or the desk,
   that part of the video cannot be located.
8. 15 to 60 seconds of video is enough. Copy the file to the computer into `data/`,
   for example `data/flyover.mp4` (`.mov` from an iPhone works too).

If you see wavy stripes (moire) on the phone screen while filming, move the phone a little
further away or a bit to the side. The program removes light moire itself.

### 2. Run it

```bash
python main.py --map data/map.png --from-video data/flyover.mp4
```

It takes 3 frames per second of video. Use `--every 5` to take every 5th video frame instead
(smaller number = more points, slower). Results go to `output/`:

- `output/video_frames/`: the video frames that were used (blurry ones are skipped)
- `output/route.png`: the route on the map, each point labelled with its time in the video
- `output/route.mp4`: zoomed map window following the drone
- `output/route_side_by_side.mp4`: the camera view (red cross = drone position) next to the
  same place on the map
- `output/route.csv`: position for every used frame

### 3. No video yet? Make a fake one

`tools/make_test_video.py` puts the map on a virtual screen and flies a virtual phone over it,
adding tilt, moire, glare, motion blur, exposure changes and noise. It also saves the true
position for every frame, so you can check the accuracy:

```bash
python tools/make_test_video.py --map data/map.png --out data/test_flyover.mp4
python main.py --map data/map.png --from-video data/test_flyover.mp4
python tools/make_test_video.py --check output/route.csv --truth data/test_flyover_truth.csv
```

On the 20 second test video all 58 used frames are found, with an error under 1 map pixel.

## How it works

1. **Features.** SIFT finds distinctive points on the map and on each frame.
2. **Matching.** Each frame's points are matched to the map's points, keeping only clear matches (Lowe's ratio test, 0.75).
3. **Position.** A homography with RANSAC gives how the frame sits on the map: position, rotation and scale. The frame's centre is the drone position and its 4 corners are the footprint.
4. **Clean-up.** Frames with too few matches are skipped. A point that jumps far from both neighbours is dropped as an outlier. Points keep their real frame numbers, so a gap stays visible.
5. **Smoothing.** A 3-point moving average removes small jitter.
6. **Drawing.** The route, points, START / FINISH and footprints are drawn on the map. The video moves a zoomed window along the interpolated route.

Very large maps are downscaled to at most 4000 px on the longest side for speed.

Extra steps for a filmed video:

- **Sampling.** Only every N-th video frame is used (3 per second by default), downscaled to 1000 px.
- **Blur check.** Frames much blurrier than a typical frame (variance of the Laplacian) are skipped.
- **Screen clean-up.** A light Gaussian blur removes moire from the screen's pixel grid, and CLAHE
  (local contrast equalisation, applied to the map too) reduces the effect of glare and dark corners.
- **Tilt.** The homography already covers a phone that is not held perfectly flat (perspective).
- **Robust fitting.** MAGSAC (a stronger RANSAC) fits the homography, and a result is rejected if too few
  matches agree, the footprint is twisted, or the zoom is absurd.
- **Smoothing.** Video points are closer together and noisier, so the moving average uses 5 points.
