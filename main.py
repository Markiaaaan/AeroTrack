"""UAV route reconstruction.

Finds where each drone frame was taken on a satellite map (SIFT features + homography),
smooths the resulting route and saves:
  * an image of the map with the route, numbered points, START / FINISH and frame footprints
  * a video where a zoomed map window follows the drone along the route

Usage:
    python main.py --map data/map.png --frames data/frames
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
MAX_MAP_SIDE = 4000      # downscale the map if it is larger than this (speed + memory)
MIN_MATCHES = 12         # fewer good matches than this -> frame counts as "not found"
OUTLIER_FACTOR = 3.0     # a point this many typical steps away from its neighbours is dropped


def read(path, flag=cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-English characters in Windows paths."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), flag)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    return img


def natural_key(path):
    """Sort key so that crop_2 comes before crop_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def parse_args():
    ap = argparse.ArgumentParser(description="Reconstruct a UAV route from frames and a map.")
    ap.add_argument("--map", default="data/map.png", help="satellite map image")
    ap.add_argument("--frames", default="data/frames", help="folder with the UAV frames")
    ap.add_argument("--out", default="output/route.png", help="output image")
    ap.add_argument("--video", default="output/route.mp4", help="output video ('' to skip)")
    return ap.parse_args()


def locate_frames(map_img, files):
    """Match every frame to the map.

    Returns a list of dicts, one per frame:
    {"name", "ok", "center": (x, y), "corners": 4x2 array, "inliers"} in map pixels.
    """
    sift = cv2.SIFT_create(nfeatures=20000)
    kp_m, des_m = sift.detectAndCompute(cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY), None)
    matcher = cv2.BFMatcher()

    results = []
    for f in files:
        name = os.path.basename(f)
        res = {"name": name, "ok": False, "center": None, "corners": None, "inliers": 0}
        results.append(res)

        img = read(f, cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(img, None)
        if des is None or len(kp) < 2:
            print(name, "- no features, skipped")
            continue

        # Lowe's ratio test keeps only matches that are clearly better than the 2nd best
        good = [m for m, n in matcher.knnMatch(des, des_m, k=2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_MATCHES:
            print(name, f"- only {len(good)} matches, skipped")
            continue

        # Homography = how the frame is moved, rotated and scaled onto the map
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print(name, "- no homography, skipped")
            continue

        # Project the frame's 4 corners and its centre onto the map
        h, w = img.shape
        pr = cv2.perspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2), H
        ).reshape(-1, 2)
        res.update(ok=True, center=pr[4], corners=pr[:4], inliers=int(mask.sum()))
        print(name, f"- ok ({res['inliers']} inliers)")
    return results


def drop_outliers(results):
    """Mark as failed any point that jumps far away from both of its neighbours."""
    idx = [i for i, r in enumerate(results) if r["ok"]]
    if len(idx) < 3:
        return
    pts = np.array([results[i]["center"] for i in idx])
    typical_step = np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    for k in range(1, len(idx) - 1):
        expected = (pts[k - 1] + pts[k + 1]) / 2          # where the point "should" be
        if np.linalg.norm(pts[k] - expected) > OUTLIER_FACTOR * typical_step:
            results[idx[k]]["ok"] = False
            print(results[idx[k]]["name"], "- outlier, dropped")


def smooth(pts):
    """Moving average over 3 points (ends are repeated so the path keeps its length)."""
    if len(pts) < 3:
        return pts
    pad = np.pad(pts, ((1, 1), (0, 0)), "edge")
    return np.stack([np.convolve(pad[:, i], np.ones(3) / 3, "valid") for i in range(2)], 1)


def draw_legend(img):
    items = [((255, 200, 0), "frame footprint"), ((0, 0, 255), "route"),
             ((0, 255, 0), "frame position (number = frame)"),
             ((255, 0, 0), "START"), ((0, 255, 255), "FINISH")]
    x, y = 20, 20
    cv2.rectangle(img, (x, y), (x + 520, y + 40 * len(items) + 20), (0, 0, 0), -1)
    for i, (color, text) in enumerate(items):
        cy = y + 35 + 40 * i
        cv2.circle(img, (x + 25, cy), 10, color, -1)
        cv2.putText(img, text, (x + 50, cy + 10), FONT, 0.9, (255, 255, 255), 2)


def draw_route(map_img, results, path, numbers):
    img = map_img.copy()
    cv2.polylines(img, [np.int32(r["corners"]) for r in results if r["ok"]], True, (255, 200, 0), 2)
    cv2.polylines(img, [np.int32(path)], False, (0, 0, 255), 4)
    for p, n in zip(path, numbers):
        x, y = map(int, p)
        cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img, str(n), (x + 10, y - 10), FONT, 1, (255, 255, 255), 2)
    for p, label, color in [(path[0], "START", (255, 0, 0)), (path[-1], "FINISH", (0, 255, 255))]:
        x, y = map(int, p)
        cv2.circle(img, (x, y), 16, color, 3)
        cv2.putText(img, label, (x + 20, y + 30), FONT, 1.2, color, 3)
    draw_legend(img)
    return img


def make_video(img, path, out, view=(800, 450), size=(1280, 720), steps=10):
    """Zoomed map window that follows the drone; the route flown so far is drawn in yellow."""
    map_h, map_w = img.shape[:2]
    vw, vh = min(view[0], map_w), min(view[1], map_h)
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, size)
    flown = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):   # interpolate between points
            p = a + (b - a) * t
            flown.append(p)
            x0 = int(np.clip(p[0] - vw / 2, 0, map_w - vw))
            y0 = int(np.clip(p[1] - vh / 2, 0, map_h - vh))
            fr = img[y0:y0 + vh, x0:x0 + vw].copy()
            cv2.polylines(fr, [np.int32(np.array(flown) - (x0, y0))], False, (0, 255, 255), 3)
            cv2.circle(fr, (int(p[0] - x0), int(p[1] - y0)), 10, (0, 0, 255), -1)
            video.write(cv2.resize(fr, size))
    video.release()


def main():
    args = parse_args()

    # 1. Load the map (downscaled if huge) and the frames in natural order
    map_img = read(args.map)
    s = min(1, MAX_MAP_SIDE / max(map_img.shape[:2]))
    map_img = cv2.resize(map_img, None, fx=s, fy=s)
    files = sorted(glob.glob(os.path.join(args.frames, "*")), key=natural_key)
    if not files:
        raise SystemExit(f"No frames found in {args.frames}")

    # 2. Localise every frame on the map, then remove obviously wrong positions
    results = locate_frames(map_img, files)
    drop_outliers(results)
    found = [(i + 1, r) for i, r in enumerate(results) if r["ok"]]
    print(f"Located {len(found)} of {len(results)} frames")
    if len(found) < 2:
        raise SystemExit("Too few frames located to draw a route")

    # 3. Smooth the route; points keep their real frame numbers even if some were skipped
    numbers = [n for n, _ in found]
    path = smooth(np.array([r["center"] for _, r in found]))

    # 4. Route image
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img = draw_route(map_img, results, path, numbers)
    cv2.imencode(os.path.splitext(args.out)[1] or ".png", img)[1].tofile(args.out)
    print("Saved", args.out)

    # 5. Optional video
    if args.video:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        make_video(img, path, args.video)
        print("Saved", args.video)


if __name__ == "__main__":
    main()
