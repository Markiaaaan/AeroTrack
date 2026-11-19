"""Make a fake "phone films the screen" flyover video, to test main.py without recording one.

It puts the map on a virtual 1920x1080 screen, flies a virtual camera over it and adds the
problems a real recording has: screen pixel grid (moire), tilt (perspective), glare,
motion blur, brightness changes and sensor noise. It also writes the true camera position
for every video frame, so you can measure how accurate main.py is.

Make the video:
    python tools/make_test_video.py --map data/map.png --out data/test_flyover.mp4
Run the tracker on it:
    python main.py --map data/map.png --from-video data/test_flyover.mp4
Compare with the truth:
    python tools/make_test_video.py --check output/route.csv --truth data/test_flyover_truth.csv
"""
import argparse
import csv
import os

import cv2
import numpy as np

SCREEN = (1920, 1080)   # virtual monitor
CAMERA = (1280, 720)    # virtual phone video
FPS = 30


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", default="data/map.png")
    ap.add_argument("--out", default="data/test_flyover.mp4")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", help="route.csv from main.py: compare it with --truth and exit")
    ap.add_argument("--truth", help="truth CSV written next to the video")
    return ap.parse_args()


def map_on_screen(map_img):
    """Fit the map into the screen (black bars), like an image viewer in full screen mode.

    Returns the screen image plus the scale and offset to go from screen to map pixels.
    """
    sw, sh = SCREEN
    s = min(sw / map_img.shape[1], sh / map_img.shape[0])
    small = cv2.resize(map_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    screen = np.zeros((sh, sw, 3), np.uint8)
    ox, oy = (sw - small.shape[1]) // 2, (sh - small.shape[0]) // 2
    screen[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    # Upscale 3x and darken the gaps between pixels: this pixel grid causes moire when filmed
    big = cv2.resize(screen, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    big[2::3, :] *= 0.6
    big[:, 2::3] *= 0.6
    return big.astype(np.uint8), s, np.array([ox, oy])


def flight_path(t, area):
    """Smooth loop over the screen (t in 0..1). area = (x0, y0, x1, y1) in screen pixels."""
    x0, y0, x1, y1 = area
    cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    a = 2 * np.pi * t
    return np.array([cx + 0.75 * rx * np.sin(a), cy + 0.6 * ry * np.sin(2 * a + 0.4)])


def camera_quad(center, heading, width, tilt, rng):
    """Where the 4 camera corners land on the screen: rotated rectangle + perspective tilt."""
    w, h = width, width * CAMERA[1] / CAMERA[0]
    # A tilted phone sees the far edge wider than the near edge (keystone effect)
    far, near = 1 + tilt, 1 - tilt
    pts = np.array([[-w / 2 * far, -h / 2], [w / 2 * far, -h / 2],
                    [w / 2 * near, h / 2], [-w / 2 * near, h / 2]])
    pts += rng.normal(0, w * 0.01, pts.shape)               # hand shake
    c, s = np.cos(heading), np.sin(heading)
    return pts @ np.array([[c, s], [-s, c]]) + center


def add_glare(img, pos, radius, strength):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2) / (2 * radius ** 2))
    out = img.astype(np.float32) + 255 * strength * blob[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1), (length, length))
    return cv2.filter2D(img, -1, k / k.sum())


def make_video(args):
    rng = np.random.default_rng(args.seed)
    map_img = cv2.imdecode(np.fromfile(args.map, np.uint8), cv2.IMREAD_COLOR)
    big, s, offset = map_on_screen(map_img)
    area = (offset[0], offset[1], SCREEN[0] - offset[0], SCREEN[1] - offset[1])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    video = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, CAMERA)
    truth_path = os.path.splitext(args.out)[0] + "_truth.csv"
    truth = csv.writer(open(truth_path, "w", newline=""))
    truth.writerow(["frame", "x", "y"])

    n = int(args.seconds * FPS)
    cam_rect = np.float32([[0, 0], [CAMERA[0], 0], [CAMERA[0], CAMERA[1]], [0, CAMERA[1]]])
    for i in range(n):
        t = i / n
        center = flight_path(t, area)
        velocity = flight_path(t + 1e-3, area) - center
        heading = np.arctan2(velocity[1], velocity[0]) + np.pi / 2   # camera "up" = flight direction
        width = 650 + 120 * np.sin(4 * np.pi * t)                     # phone moves closer / further
        tilt = 0.12 + 0.08 * np.sin(6 * np.pi * t)
        quad = camera_quad(center, heading, width, tilt, rng)

        # Screen (3x upscaled) -> camera image
        H = cv2.getPerspectiveTransform(np.float32(quad * 3), cam_rect)
        frame = cv2.warpPerspective(big, H, CAMERA, flags=cv2.INTER_LINEAR)

        # Recording problems
        if i % 45 < 6:                                     # a quick jerk every 1.5 s
            frame = motion_blur(frame, 15, np.degrees(heading))
        frame = cv2.GaussianBlur(frame, (3, 3), 0.8)       # lens is never perfectly sharp
        glare_pos = (CAMERA[0] * (0.3 + 0.4 * np.sin(2 * np.pi * t)), CAMERA[1] * 0.35)
        frame = add_glare(frame, glare_pos, 160, 0.45)
        gain = 0.85 + 0.15 * np.sin(2 * np.pi * 3 * t)      # auto-exposure pumping
        frame = np.clip(frame * gain + rng.normal(0, 6, frame.shape), 0, 255).astype(np.uint8)
        video.write(frame)

        # True position: camera image centre -> screen -> map pixels
        cx, cy = cv2.perspectiveTransform(
            np.float32([[[CAMERA[0] / 2, CAMERA[1] / 2]]]), np.linalg.inv(H))[0, 0] / 3
        mx, my = (np.array([cx, cy]) - offset) / s
        truth.writerow([i, f"{mx:.1f}", f"{my:.1f}"])
    video.release()
    print(f"Saved {args.out} ({n} frames) and {truth_path}")


def check(route_csv, truth_csv):
    """Print how far main.py's positions are from the true ones, in map pixels."""
    truth = {int(r["frame"]): (float(r["x"]), float(r["y"])) for r in csv.DictReader(open(truth_csv))}
    errors, missed = [], 0
    for r in csv.DictReader(open(route_csv)):
        if r["found"] != "1":
            missed += 1
            continue
        index = int(r["frame"].split("_")[-1])
        tx, ty = truth[index]
        errors.append(np.hypot(float(r["x"]) - tx, float(r["y"]) - ty))
    errors = np.array(errors)
    print(f"Located {len(errors)} frames, missed {missed}")
    print(f"Error in map pixels: median {np.median(errors):.1f}, "
          f"90% of frames under {np.percentile(errors, 90):.1f}, worst {errors.max():.1f}")


if __name__ == "__main__":
    a = parse_args()
    if a.check:
        check(a.check, a.truth or os.path.splitext(a.out)[0] + "_truth.csv")
    else:
        make_video(a)
