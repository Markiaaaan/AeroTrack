"""UAV route reconstruction.

Finds where each drone frame was taken on a satellite map (SIFT features + homography),
smooths the resulting route and saves:
  * an image of the map with the route, numbered points, START / FINISH and frame footprints
  * a video where a zoomed map window follows the drone along the route
  * a CSV file with the position of every frame

Two kinds of input:
  * a folder of frames (basic level):
        python main.py --map data/map.png --frames data/frames
  * a video of the map filmed from a screen with a phone or webcam (advanced level):
        python main.py --map data/map.png --from-video data/flyover.mp4
    For a video it also saves a side-by-side video: camera on the left, map position on the right.
"""
import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_MAP_SIDE = 4000      # downscale the map if it is larger than this (speed + memory)
MAX_FRAME_SIDE = 1000    # downscale video frames to this; phone 4K frames only add noise and time
MIN_MATCHES = 12         # fewer good matches than this -> frame counts as "not found"
MIN_INLIERS = 8          # fewer matches that agree with the homography -> "not found"
MIN_INLIER_RATIO = 0.15  # less than this share of matches agree -> the match is probably wrong
BLUR_RATIO = 0.35        # a video frame this much blurrier than a typical one is skipped
OUTLIER_FACTOR = 3.0     # a point this many typical steps away from its neighbours is dropped
MAX_LABELS = 60          # with more points than this only every k-th point gets a number
MAX_VIDEO_FOOTPRINTS = 10  # a filmed frame covers a big area, so draw only a few footprints

# MAGSAC is a newer, more robust version of RANSAC (OpenCV >= 4.5). Fall back to RANSAC if missing.
ROBUST_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def read(path, flag=cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-English characters in Windows paths."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), flag)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    return img


def write(path, img):
    """cv2.imwrite that also works with non-English characters in Windows paths."""
    cv2.imencode(os.path.splitext(path)[1] or ".png", img)[1].tofile(path)


def natural_key(path):
    """Sort key so that crop_2 comes before crop_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def parse_args():
    ap = argparse.ArgumentParser(description="Reconstruct a UAV route from frames and a map.")
    ap.add_argument("--map", default="data/map.png", help="satellite map image")
    ap.add_argument("--frames", default="data/frames", help="folder with the UAV frames")
    ap.add_argument("--from-video", default="",
                    help="video of the map filmed from a screen (used instead of --frames)")
    ap.add_argument("--every", type=int, default=0,
                    help="with --from-video: use every N-th video frame (default: 3 per second)")
    ap.add_argument("--out", default="output/route.png", help="output image")
    ap.add_argument("--video", default="output/route.mp4", help="output video ('' to skip)")
    return ap.parse_args()


# ---------------------------------------------------------------- input frames

def frames_from_folder(folder):
    """Load every image in a folder (natural order). Returns a list of frame dicts."""
    files = sorted(glob.glob(os.path.join(folder, "*")), key=natural_key)
    if not files:
        raise SystemExit(f"No frames found in {folder}")
    return [{"name": os.path.basename(f), "label": str(i + 1), "image": read(f)}
            for i, f in enumerate(files)]


def sharpness(gray):
    """Variance of the Laplacian: high for sharp images, low for blurry ones."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frames_from_video(path, every, save_dir):
    """Take every N-th frame of a video, drop the blurriest ones and save the rest as JPEGs.

    Returns a list of frame dicts, like frames_from_folder.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    every = every or max(1, round(fps / 3))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames at {fps:.0f} fps, using every {every}-th frame")

    frames, index = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if index % every == 0:
            s = min(1, MAX_FRAME_SIDE / max(img.shape[:2]))
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            frames.append({"name": f"frame_{index:05d}", "label": f"{index / fps:.1f}s",
                           "index": index, "image": img,
                           "sharpness": sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))})
        index += 1
    cap.release()
    if not frames:
        raise SystemExit(f"No frames could be read from {path}")

    # Motion blur and refocusing make some frames useless; compare each to the typical sharpness
    typical = np.median([f["sharpness"] for f in frames])
    kept = [f for f in frames if f["sharpness"] >= BLUR_RATIO * typical]
    print(f"Skipped {len(frames) - len(kept)} blurry frames, {len(kept)} left")

    os.makedirs(save_dir, exist_ok=True)
    for f in kept:
        write(os.path.join(save_dir, f["name"] + ".jpg"), f["image"])
    print("Extracted frames saved to", save_dir)
    return kept


def clean_screen_photo(gray):
    """Make a photo of a screen look more like the original map.

    * a light blur removes moire (wavy stripes from the screen's pixel grid)
    * CLAHE evens out contrast, so glare and dark corners hurt less
    """
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


# ---------------------------------------------------------------- matching

def good_homography(H, corners, w, h, inliers, n_matches):
    """Reject homographies that cannot be a real camera view of the map."""
    if H is None or inliers < MIN_INLIERS or inliers < MIN_INLIER_RATIO * n_matches:
        return False
    quad = np.float32(corners).reshape(-1, 1, 2)
    if not cv2.isContourConvex(quad):           # a real view is never twisted or folded
        return False
    scale = np.sqrt(cv2.contourArea(quad) / (w * h))
    return 0.05 < scale < 20                     # absurd zoom = wrong match


def locate_frames(map_gray, frames, screen_video):
    """Match every frame to the map.

    Adds to every frame dict: "ok", "center": (x, y), "corners": 4x2 array, "inliers",
    all in map pixels.
    """
    sift = cv2.SIFT_create(nfeatures=20000)
    kp_m, des_m = sift.detectAndCompute(map_gray, None)
    matcher = cv2.BFMatcher()

    for fr in frames:
        fr.update(ok=False, center=None, corners=None, inliers=0)
        name = fr["name"]
        img = cv2.cvtColor(fr["image"], cv2.COLOR_BGR2GRAY)
        if screen_video:
            img = clean_screen_photo(img)
        kp, des = sift.detectAndCompute(img, None)
        if des is None or len(kp) < 2:
            print(name, "- no features, skipped")
            continue

        # Lowe's ratio test keeps only matches that are clearly better than the 2nd best
        good = [m for m, n in matcher.knnMatch(des, des_m, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_MATCHES:
            print(name, f"- only {len(good)} matches, skipped")
            continue

        # Homography = how the frame is moved, rotated, scaled and tilted onto the map
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, ROBUST_METHOD, 5.0, maxIters=5000, confidence=0.999)
        if H is None:
            print(name, "- no homography, skipped")
            continue

        # Project the frame's 4 corners and its centre onto the map
        h, w = img.shape
        pr = cv2.perspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        inliers = int(mask.sum())
        if not good_homography(H, pr[:4], w, h, inliers, len(good)):
            print(name, f"- unreliable match ({inliers}/{len(good)} inliers), skipped")
            continue
        fr.update(ok=True, center=pr[4], corners=pr[:4], inliers=inliers)
        print(name, f"- ok ({inliers} inliers)")


def drop_outliers(frames):
    """Mark as failed any point that jumps far away from both of its neighbours."""
    idx = [i for i, r in enumerate(frames) if r["ok"]]
    if len(idx) < 3:
        return
    pts = np.array([frames[i]["center"] for i in idx])
    # at least 10 px, so a drone that hovers in place does not make every wobble an "outlier"
    typical_step = max(10.0, np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    for k in range(1, len(idx) - 1):
        expected = (pts[k - 1] + pts[k + 1]) / 2          # where the point "should" be
        if np.linalg.norm(pts[k] - expected) > OUTLIER_FACTOR * typical_step:
            frames[idx[k]]["ok"] = False
            print(frames[idx[k]]["name"], "- outlier, dropped")


def smooth(pts, window=3):
    """Moving average (ends are repeated so the path keeps its length)."""
    if len(pts) < window:
        return pts
    pad = np.pad(pts, ((window // 2, window // 2), (0, 0)), "edge")
    return np.stack([np.convolve(pad[:, i], np.ones(window) / window, "valid") for i in range(2)], 1)


# ---------------------------------------------------------------- drawing

def draw_legend(img, number_text):
    items = [((255, 200, 0), "frame footprint"), ((0, 0, 255), "route"),
             ((0, 255, 0), f"frame position ({number_text})"),
             ((255, 0, 0), "START"), ((0, 255, 255), "FINISH")]
    x, y = 20, 20
    cv2.rectangle(img, (x, y), (x + 560, y + 40 * len(items) + 20), (0, 0, 0), -1)
    for i, (color, text) in enumerate(items):
        cy = y + 35 + 40 * i
        cv2.circle(img, (x + 25, cy), 10, color, -1)
        cv2.putText(img, text, (x + 50, cy + 10), FONT, 0.9, (255, 255, 255), 2)


def draw_route(map_img, found, path, number_text, max_footprints=None):
    """found: located frame dicts in order; path: their smoothed centres."""
    img = map_img.copy()
    k = max(1, int(np.ceil(len(found) / MAX_LABELS)))   # label every k-th point
    kf = max(k, int(np.ceil(len(found) / max_footprints))) if max_footprints else k
    cv2.polylines(img, [np.int32(f["corners"]) for f in found[::kf]], True, (255, 200, 0), 2)
    cv2.polylines(img, [np.int32(path)], False, (0, 0, 255), 4)
    for i, (p, f) in enumerate(zip(path, found)):
        x, y = map(int, p)
        cv2.circle(img, (x, y), 8 if i % k == 0 else 4, (0, 255, 0), -1)
        if i % k == 0:
            cv2.putText(img, f["label"], (x + 10, y - 10), FONT, 1, (255, 255, 255), 2)
    for p, label, color in [(path[0], "START", (255, 0, 0)), (path[-1], "FINISH", (0, 255, 255))]:
        x, y = map(int, p)
        cv2.circle(img, (x, y), 16, color, 3)
        cv2.putText(img, label, (x + 20, y + 30), FONT, 1.2, color, 3)
    draw_legend(img, number_text)
    return img


def crop_around(img, p, vw, vh):
    """Window of size vw x vh centred on point p (shifted to stay inside the image)."""
    h, w = img.shape[:2]
    x0 = int(np.clip(p[0] - vw / 2, 0, w - vw))
    y0 = int(np.clip(p[1] - vh / 2, 0, h - vh))
    return img[y0:y0 + vh, x0:x0 + vw].copy(), np.array([x0, y0])


def make_video(img, path, out, view=(800, 450), size=(1280, 720), steps=10):
    """Zoomed map window that follows the drone; the route flown so far is drawn in yellow."""
    vw, vh = min(view[0], img.shape[1]), min(view[1], img.shape[0])
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    flown = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):   # interpolate between points
            p = a + (b - a) * t
            flown.append(p)
            fr, origin = crop_around(img, p, vw, vh)
            cv2.polylines(fr, [np.int32(np.array(flown) - origin)], False, (0, 255, 255), 3)
            cv2.circle(fr, tuple(np.int32(p - origin)), 10, (0, 0, 255), -1)
            video.write(cv2.resize(fr, size))
    video.release()


def make_side_by_side(img, found, path, out, fps, half=(640, 480)):
    """Left: what the camera saw. Right: where that is on the map (footprint + route so far).

    img should be the clean map, so only the route flown so far is drawn on it.
    """
    # Window size: big enough to show a typical footprint with some map around it
    span = np.median([np.ptp(f["corners"], axis=0).max() for f in found])
    vw = int(min(img.shape[1], img.shape[0] * half[0] / half[1], 1.6 * span))
    vh = int(vw * half[1] / half[0])
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * half[0], half[1]))
    for i, (f, p) in enumerate(zip(found, path)):
        cam = f["image"]
        s = min(half[0] / cam.shape[1], half[1] / cam.shape[0])
        cam = cv2.resize(cam, None, fx=s, fy=s)
        left = np.zeros((half[1], half[0], 3), np.uint8)
        y, x = (half[1] - cam.shape[0]) // 2, (half[0] - cam.shape[1]) // 2
        left[y:y + cam.shape[0], x:x + cam.shape[1]] = cam
        cx, cy = half[0] // 2, half[1] // 2                 # camera centre = drone position
        cv2.drawMarker(left, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(left, f"camera {f['label']}", (10, 30), FONT, 0.8, (0, 255, 255), 2)

        right, origin = crop_around(img, p, vw, vh)
        cv2.polylines(right, [np.int32(path[:i + 1] - origin)], False, (0, 255, 255), 4)
        cv2.polylines(right, [np.int32(f["corners"] - origin)], True, (255, 200, 0), 4)
        cv2.circle(right, tuple(np.int32(p - origin)), 12, (0, 0, 255), -1)
        right = cv2.resize(right, half)
        cv2.putText(right, "position on map", (10, 30), FONT, 0.8, (0, 255, 255), 2)
        video.write(np.hstack([left, right]))
    video.release()


def save_csv(path, frames):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "label", "found", "x", "y", "inliers"])
        for f in frames:
            x, y = f["center"] if f["ok"] else ("", "")
            w.writerow([f["name"], f["label"], int(f["ok"]),
                        f"{x:.1f}" if f["ok"] else x, f"{y:.1f}" if f["ok"] else y, f["inliers"]])


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    screen_video = bool(args.from_video)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load the map (downscaled if huge) and the frames
    map_img = read(args.map)
    s = min(1, MAX_MAP_SIDE / max(map_img.shape[:2]))
    map_img = cv2.resize(map_img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    map_gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
    if screen_video:
        # CLAHE on the map too, so map and screen photos have similar contrast
        map_gray = clean_screen_photo(map_gray)
        frames = frames_from_video(args.from_video, args.every, os.path.join(out_dir, "video_frames"))
    else:
        frames = frames_from_folder(args.frames)

    # 2. Localise every frame on the map, then remove obviously wrong positions
    locate_frames(map_gray, frames, screen_video)
    drop_outliers(frames)
    found = [f for f in frames if f["ok"]]
    print(f"Located {len(found)} of {len(frames)} frames")
    if len(found) < 2:
        raise SystemExit("Too few frames located to draw a route")

    # 3. Smooth the route; points keep their own labels even if some frames were skipped.
    #    A video gives many close points with more jitter, so it gets a wider window.
    path = smooth(np.array([f["center"] for f in found]), window=5 if screen_video else 3)

    # 4. Route image and CSV with positions
    number_text = "label = seconds into video" if screen_video else "number = frame"
    img = draw_route(map_img, found, path, number_text,
                     max_footprints=MAX_VIDEO_FOOTPRINTS if screen_video else None)
    write(args.out, img)
    print("Saved", args.out)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    save_csv(csv_path, frames)
    print("Saved", csv_path)

    # 5. Optional videos
    if args.video:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        # about 20 seconds of video whatever the number of points
        make_video(img, path, args.video, steps=max(2, 500 // len(path)))
        print("Saved", args.video)
        if screen_video:
            side = os.path.splitext(args.video)[0] + "_side_by_side.mp4"
            make_side_by_side(map_img, found, path, side, fps=5)
            print("Saved", side)


if __name__ == "__main__":
    main()
