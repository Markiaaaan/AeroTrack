"""UAV route reconstruction.

Finds where each drone frame was taken on a satellite map (SIFT features + homography),
removes wrong positions and saves:
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
import itertools
import os
import re

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Only files with these extensions are read from the frames folder
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
# Downscale the map if it is larger than this (speed + memory)
MAX_MAP_SIDE = 4000
# Downscale video frames to this; phone 4K frames only add noise and time
MAX_FRAME_SIDE = 1000
# How many SIFT points to keep on the map (the best ones)
SIFT_FEATURES = 20000
# Lowe's ratio test: the best match must be clearly better than the 2nd best
RATIO_TEST = 0.75
# A match further than this many map pixels from the homography is an outlier
RANSAC_THRESHOLD = 5.0
# Frame bigger than 2x its footprint on the map -> failed frames are retried at half size
RETRY_BELOW_SCALE = 0.5
# Fewer good matches than this -> the frame counts as "not found"
MIN_MATCHES = 12
# Fewer matches that agree with the homography -> "not found"
MIN_INLIERS = 8
# Less than this share of matches agree -> the match is probably wrong
MIN_INLIER_RATIO = 0.15
# A video frame this much blurrier than a typical one is skipped
BLUR_RATIO = 0.35
# A jump to the next frame longer than this many typical steps is not a real move (outlier).
# 4 leaves room for a quick jerk of a hand-held phone (seen at about 3.5 steps in a real video)
OUTLIER_FACTOR = 4.0
# Typical step = median of the steps within this many points before and after
STEP_WINDOW = 10
# px; the typical step is at least this, so a hovering drone's wobble is not an "outlier"
MIN_STEP = 10.0
# A footprint smaller / bigger than these times the typical one is a wrong match
SIZE_RANGE = (0.5, 2.0)
# With more points than this only every k-th point gets a number
MAX_LABELS = 60
# A filmed frame covers a big area, so draw only a few footprints
MAX_VIDEO_FOOTPRINTS = 10

REFERENCE_MAP_SIDE = 3645  # drawing sizes (text, lines) are tuned for a map this wide

# Colours (B, G, R) shared by the route image, its legend and the video
COLORS = {
    "footprint": (255, 200, 0),  # light blue
    "route": (0, 0, 255),  # red
    "gap": (0, 165, 255),  # orange: dashed where frames are missing or not located
    "point": (0, 255, 0),  # green
    "start": (255, 0, 255),  # magenta
    "finish": (0, 255, 255),  # yellow
    "flown": (255, 255, 255),  # white: part of the route already flown in the video
}

# MAGSAC is a newer, more robust version of RANSAC (OpenCV >= 4.5). Fall back to RANSAC if missing.
ROBUST_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def read(path, flag=cv2.IMREAD_COLOR, required=True):
    """cv2.imread that also works with non-English characters in Windows paths.

    If the image cannot be read: stop with a message (required=True) or return None.
    """
    try:
        img = cv2.imdecode(np.fromfile(path, np.uint8), flag)
    except (OSError, cv2.error):  # missing file, a folder, empty or broken file
        img = None
    if img is None and required:
        raise SystemExit(f"Cannot read image: {path}")
    return img


def write(path, img):
    """cv2.imwrite that also works with non-English characters in Windows paths."""
    cv2.imencode(os.path.splitext(path)[1] or ".png", img)[1].tofile(path)


def natural_key(path):
    """Sort key so that crop_2 comes before crop_10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]


def first_existing(*paths):
    """First path that exists (or the first one, so the error message names it)."""
    return next((p for p in paths if os.path.exists(p)), paths[0])


def parse_args():
    ap = argparse.ArgumentParser(
        description="Reconstruct a UAV route from frames and a map."
    )
    # Defaults: data/ (as in the README), or Materials/ if that is where the files are
    ap.add_argument(
        "--map",
        default=first_existing("data/map.png", "Materials/global_map.png"),
        help="satellite map image",
    )
    ap.add_argument(
        "--frames",
        default=first_existing("data/frames", "Materials/crops"),
        help="folder with the UAV frames",
    )
    ap.add_argument(
        "--from-video",
        default="",
        help="video of the map filmed from a screen (used instead of --frames)",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=0,
        help="with --from-video: use every N-th video frame (default: 3 per second)",
    )
    ap.add_argument("--out", default="output/route.png", help="output image")
    ap.add_argument(
        "--video", default="output/route.mp4", help="output video ('' to skip)"
    )
    return ap.parse_args()


# ---------------------------------------------------------------- input frames


def frame_label(path, i):
    """Label for a frame: the number in its file name (crop_25.png -> "25"), else its position."""
    numbers = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return str(int(numbers[-1])) if numbers else str(i)


def frames_from_folder(folder):
    """Load every image in a folder (natural order). Returns a list of frame dicts.

    Other files (notes, Thumbs.db, sub-folders) are ignored; broken images are skipped.
    """
    files = sorted(
        (
            f
            for f in glob.glob(os.path.join(folder, "*"))
            if os.path.isfile(f) and f.lower().endswith(IMAGE_EXTENSIONS)
        ),
        key=natural_key,
    )
    frames = []
    for i, f in enumerate(files):
        img = read(f, required=False)
        if img is None:
            print(os.path.basename(f), "- cannot read this image, skipped")
            continue
        label = frame_label(f, i)
        # "number" is used to see gaps: crop_24 -> crop_26 means frame 25 is missing
        frames.append(
            {
                "name": os.path.basename(f),
                "label": label,
                "number": int(label),
                "image": img,
            }
        )
    if not frames:
        raise SystemExit(f"No frames found in {folder}")
    return frames


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
            frames.append(
                {
                    "name": f"frame_{index:05d}",
                    "label": f"{index / fps:.1f}s",
                    "index": index,
                    # place in the sequence of sampled frames; blurry frames dropped below
                    # still count, so a skipped stretch shows up as a longer gap in time
                    "step": len(frames),
                    "image": img,
                    "sharpness": sharpness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                }
            )
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
    if not cv2.isContourConvex(quad):  # a real view is never twisted or folded
        return False
    # A camera cannot see a mirror image: the corners must go round in the same direction
    # as in the frame (signed area keeps its sign)
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    if cv2.contourArea(quad, oriented=True) * cv2.contourArea(rect, oriented=True) <= 0:
        return False
    scale = np.sqrt(cv2.contourArea(quad) / (w * h))
    return 0.05 < scale < 20  # absurd zoom = wrong match


def locate_frames(map_gray, frames, screen_video):
    """Match every frame to the map.

    Adds to every frame dict: "ok", "center": (x, y), "corners": 4x2 array, "inliers",
    all in map pixels.
    """
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp_m, des_m = sift.detectAndCompute(map_gray, None)
    matcher = cv2.BFMatcher()

    def match(fr, img):
        """Find one frame (gray image img) on the map. Returns True if found."""
        name = fr["name"]
        kp, des = sift.detectAndCompute(img, None)
        if des is None or len(kp) < 2:
            print(name, "- no features, skipped")
            return False

        # Lowe's ratio test keeps only matches that are clearly better than the 2nd best
        good = [
            m
            for m, n in matcher.knnMatch(des, des_m, k=2)
            if m.distance < RATIO_TEST * n.distance
        ]
        if len(good) < MIN_MATCHES:
            print(name, f"- only {len(good)} matches, skipped")
            return False

        # Homography = how the frame is moved, rotated, scaled and tilted onto the map
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(
            src, dst, ROBUST_METHOD, RANSAC_THRESHOLD, maxIters=5000, confidence=0.999
        )
        if H is None:
            print(name, "- no homography, skipped")
            return False

        # Project the frame's 4 corners and its centre onto the map
        h, w = img.shape
        pr = cv2.perspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(
                -1, 1, 2
            ),
            H,
        ).reshape(-1, 2)
        inliers = int(mask.sum())
        if not good_homography(H, pr[:4], w, h, inliers, len(good)):
            print(name, f"- unreliable match ({inliers}/{len(good)} inliers), skipped")
            return False
        fr.update(ok=True, center=pr[4], corners=pr[:4], inliers=inliers)
        print(name, f"- ok ({inliers} inliers)")
        return True

    def gray_of(fr):
        img = cv2.cvtColor(fr["image"], cv2.COLOR_BGR2GRAY)
        return clean_screen_photo(img) if screen_video else img

    for fr in frames:
        fr.update(ok=False, center=None, corners=None, inliers=0)
        match(fr, gray_of(fr))

    # Second try at half size for the frames that failed, if frames turned out to be much
    # bigger than their footprint on the map (a frame pixel covers less than half a map pixel).
    # Shrinking such a frame makes its details look more like the map's details.
    found = [f for f in frames if f["ok"]]
    failed = [f for f in frames if not f["ok"]]
    if found and failed:
        scale = np.median(
            [
                footprint_side(f) / np.sqrt(f["image"].shape[0] * f["image"].shape[1])
                for f in found
            ]
        )
        if scale < RETRY_BELOW_SCALE:
            for fr in failed:
                print(fr["name"], "- trying again at half size")
                match(
                    fr,
                    cv2.resize(
                        gray_of(fr), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                    ),
                )


def footprint_side(frame):
    """Size of a frame's footprint on the map: square root of its area, in map pixels."""
    return np.sqrt(cv2.contourArea(np.float32(frame["corners"])))


def drop_wrong_sizes(frames):
    """Mark as failed any frame whose footprint is much smaller or bigger than the typical one.

    All frames are taken from about the same height, so their footprints have about the same
    size. A wrong match often squeezes the frame into a few pixels or blows it up.
    """
    found = [f for f in frames if f["ok"]]
    if len(found) < 3:
        return
    typical = np.median([footprint_side(f) for f in found])
    for f in found:
        ratio = footprint_side(f) / typical
        if not SIZE_RANGE[0] <= ratio <= SIZE_RANGE[1]:
            f["ok"] = False
            print(f["name"], f"- footprint {ratio:.2f}x the typical size, dropped")


def drop_outliers(frames):
    """Mark as failed any point that does not fit into the route.

    The route is built as a chain: start at some point and walk forwards (then backwards)
    through the sequence. A point joins the chain if the drone could have flown there from
    the last point of the chain: at most one typical step per frame in between, plus
    OUTLIER_FACTOR - 1 steps of slack. A point that does not fit is skipped, and the walk goes
    on from the last good point, so one wrong match never hides its good neighbours.

    Every point is tried as the start; the longest chain is the route and the points that are
    not in it are wrong matches. The typical step is measured locally (around each point), so
    a drone or phone that speeds up or slows down is judged by its own speed there.
    """
    idx = [i for i, r in enumerate(frames) if r["ok"]]
    # With fewer than 4 points nothing can be judged reliably: there is no "majority"
    if len(idx) < 4:
        return
    pts = np.array([frames[i]["center"] for i in idx])
    # Position of every point in the sequence (file number, video sample, or place in the
    # list); a missing crop_25 makes crop_24 -> crop_26 two steps apart, and skipped blurry
    # video frames make the drone fly further between two kept ones
    seq = np.array([frames[i].get("number", frames[i].get("step", i)) for i in idx])
    # Length of each step, per frame (a jump over a missing frame counts as 2 steps)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.maximum(1, np.diff(seq))
    # Typical step near each point: the median of the 2 * STEP_WINDOW steps around it. The
    # median ignores the few long jumps to and from wrong matches. Near the start and the end
    # the window is moved inwards instead of cut short, so it still holds enough good steps.
    size = min(len(steps), 2 * STEP_WINDOW)
    local = []
    for k in range(len(pts)):
        first = int(np.clip(k - STEP_WINDOW, 0, len(steps) - size))
        local.append(max(MIN_STEP, np.median(steps[first : first + size])))

    def chain_from(start):
        """Indexes of the points that fit into a chain that goes through point start."""
        chain = [start]
        for direction in (1, -1):  # forwards, then backwards
            last = start
            end = len(pts) if direction == 1 else -1
            for k in range(start + direction, end, direction):
                frames_apart = abs(seq[k] - seq[last])
                max_jump = local[last] * (frames_apart + OUTLIER_FACTOR - 1)
                if np.linalg.norm(pts[k] - pts[last]) <= max_jump:
                    chain.append(k)
                    last = k
        return set(chain)

    # The longest chain is the route; points that do not fit into it are wrong matches
    route = max((chain_from(start) for start in range(len(pts))), key=len)
    for k in range(len(pts)):
        if k not in route:
            frames[idx[k]]["ok"] = False
            print(frames[idx[k]]["name"], "- does not fit the route (outlier), dropped")


def smooth(pts, window):
    """Moving average over `window` points that removes small jitter (used for video).

    Near the ends the window shrinks (it stays centred on the point), so the first and last
    points are never moved: START and FINISH stay where the drone really was.
    """
    half = window // 2
    out = np.array(pts, dtype=float)
    for i in range(len(pts)):
        r = min(half, i, len(pts) - 1 - i)
        out[i] = np.mean(pts[i - r : i + r + 1], axis=0)
    return out


# ---------------------------------------------------------------- drawing


def put_text(img, text, org, scale, color, thickness):
    """Text with a thin black outline, so it stays readable on any part of the map."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def dashed_line(img, a, b, color, thickness, dash):
    """Draw a dashed line from a to b (dash = length of one dash in pixels)."""
    a, b = np.float32(a), np.float32(b)
    n = max(1, int(np.linalg.norm(b - a) / dash))
    for i in range(0, n, 2):  # every second piece is drawn
        p, q = a + (b - a) * i / n, a + (b - a) * min(i + 1, n) / n
        cv2.line(
            img, tuple(np.int32(p)), tuple(np.int32(q)), color, thickness, cv2.LINE_AA
        )


def draw_legend(img, number_text, avoid, has_gap, s):
    """Draw the legend box in the image corner that is farthest from the route.

    img: image to draw on; number_text: what the point numbers mean; avoid: Nx2 route
    points the box should not cover; has_gap: also explain the dashed "gap" line;
    s: size factor (1.0 for a map REFERENCE_MAP_SIDE pixels wide).
    """
    items = [("footprint", "frame footprint"), ("route", "route (in frame order)")]
    if has_gap:
        items.append(("gap", "gap: frames missing or not located"))
    items += [
        ("point", f"frame position ({number_text})"),
        ("start", "START"),
        ("finish", "FINISH"),
    ]

    font, thick = 0.9 * s, max(1, round(2 * s))
    row, pad = int(40 * s), int(20 * s)
    text_w = max(cv2.getTextSize(t, FONT, font, thick)[0][0] for _, t in items)
    bw, bh = int(70 * s) + text_w + pad, row * len(items) + pad
    h, w = img.shape[:2]

    # Try the 4 corners; keep the one whose box is farthest from every route point
    def distance_to_route(x0, y0):
        dx = np.maximum(0, np.maximum(x0 - avoid[:, 0], avoid[:, 0] - (x0 + bw)))
        dy = np.maximum(0, np.maximum(y0 - avoid[:, 1], avoid[:, 1] - (y0 + bh)))
        return np.hypot(dx, dy).min()

    corners = [
        (pad, pad),
        (w - bw - pad, pad),
        (pad, h - bh - pad),
        (w - bw - pad, h - bh - pad),
    ]
    x, y = max(corners, key=lambda c: distance_to_route(*c))

    box = img[max(0, y) : y + bh, max(0, x) : x + bw]
    box[:] = box // 4  # darkened, but the map still shows through
    r = int(10 * s)
    for i, (key, text) in enumerate(items):
        cx, cy = x + int(35 * s), y + pad + row * i + row // 2 - int(5 * s)
        color = COLORS[key]
        if key == "footprint":
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thick)
        elif key == "route":
            cv2.line(img, (cx - 2 * r, cy), (cx + 2 * r, cy), color, thick * 2)
        elif key == "gap":
            dashed_line(img, (cx - 2 * r, cy), (cx + 2 * r, cy), color, thick * 2, r)
        elif key == "point":
            cv2.circle(img, (cx, cy), r, color, -1)
        else:
            cv2.circle(img, (cx, cy), r, color, thick)
        put_text(
            img,
            text,
            (cx + int(35 * s), cy + int(10 * s)),
            font,
            (255, 255, 255),
            thick,
        )


def draw_route(map_img, found, path, number_text, max_footprints=None):
    """Draw the route on a copy of the map and return it.

    found: located frame dicts in route order; path: the point drawn for each of them
    (Nx2, map pixels); number_text: legend text saying what the point labels mean;
    max_footprints: draw at most this many footprints (None = all).
    Where frames are missing between two points (crop_24 -> crop_26) the line is dashed.
    """
    img = map_img.copy()
    s = max(0.4, max(img.shape[:2]) / REFERENCE_MAP_SIDE)  # line and text size factor
    thin, thick = max(1, round(2 * s)), max(1, round(4 * s))

    # Footprints: thin and half transparent, so they do not hide the route
    k = max(1, int(np.ceil(len(found) / MAX_LABELS)))  # label every k-th point
    kf = max(k, int(np.ceil(len(found) / max_footprints))) if max_footprints else k
    overlay = img.copy()
    cv2.polylines(
        overlay,
        [np.int32(f["corners"]) for f in found[::kf]],
        True,
        COLORS["footprint"],
        thin,
        cv2.LINE_AA,
    )
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # Route: solid between consecutive frames, dashed where frames are missing in between
    has_gap = False
    for i in range(1, len(path)):
        gap = "number" in found[i] and found[i]["number"] - found[i - 1]["number"] > 1
        if gap:
            has_gap = True
            dashed_line(img, path[i - 1], path[i], COLORS["gap"], thick, int(15 * s))
        else:
            cv2.line(
                img,
                tuple(np.int32(path[i - 1])),
                tuple(np.int32(path[i])),
                COLORS["route"],
                thick,
                cv2.LINE_AA,
            )

    # Points with their labels (frame number from the file name)
    for i, (p, f) in enumerate(zip(path, found)):
        x, y = map(int, p)
        cv2.circle(
            img,
            (x, y),
            int((8 if i % k == 0 else 4) * s),
            COLORS["point"],
            -1,
            cv2.LINE_AA,
        )
        if i % k == 0:
            put_text(
                img,
                f["label"],
                (x + int(10 * s), y - int(10 * s)),
                s,
                (255, 255, 255),
                thin,
            )

    # START and FINISH: a ring around the true first / last position, text kept inside the image
    for p, label, key in [(path[0], "START", "start"), (path[-1], "FINISH", "finish")]:
        x, y = map(int, p)
        cv2.circle(
            img, (x, y), int(16 * s), COLORS[key], max(1, round(3 * s)), cv2.LINE_AA
        )
        tw, th = cv2.getTextSize(label, FONT, 1.2 * s, max(1, round(3 * s)))[0]
        tx = int(np.clip(x + 20 * s, 0, img.shape[1] - tw))
        ty = int(np.clip(y + 30 * s + th / 2, th, img.shape[0] - 5))
        put_text(img, label, (tx, ty), 1.2 * s, COLORS[key], max(1, round(3 * s)))

    draw_legend(img, number_text, np.float32(path), has_gap, s)
    return img


def crop_around(img, p, vw, vh):
    """Window of size vw x vh centred on point p (shifted to stay inside the image)."""
    h, w = img.shape[:2]
    x0 = int(np.clip(p[0] - vw / 2, 0, w - vw))
    y0 = int(np.clip(p[1] - vh / 2, 0, h - vh))
    return img[y0 : y0 + vh, x0 : x0 + vw].copy(), np.array([x0, y0])


def open_video(out, fps, size):
    """H.264 plays in VS Code and browsers; mp4v is the fallback when no H.264 encoder exists."""
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not video.isOpened():
        video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return video


def make_video(img, path, out, view=(800, 450), size=(1280, 720), steps=10):
    """Zoomed map window that follows the drone; the route flown so far is drawn in white."""
    vw, vh = min(view[0], img.shape[1]), min(view[1], img.shape[0])
    video = open_video(out, 25, size)
    # Points between route points (interpolation), then the last point itself, so the drone
    # really reaches FINISH; the last position is held for one second (25 frames)
    positions = [
        a + (b - a) * t
        for a, b in itertools.pairwise(path)
        for t in np.linspace(0, 1, steps, endpoint=False)
    ]
    positions += [path[-1]] * 25
    flown = [path[0]]
    for p in positions:
        flown.append(p)
        fr, origin = crop_around(img, p, vw, vh)
        cv2.polylines(
            fr, [np.int32(np.array(flown) - origin)], False, COLORS["flown"], 3
        )
        cv2.circle(fr, tuple(np.int32(p - origin)), 10, COLORS["route"], -1)
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
    video = open_video(out, fps, (2 * half[0], half[1]))
    for i, (f, p) in enumerate(zip(found, path)):
        cam = f["image"]
        s = min(half[0] / cam.shape[1], half[1] / cam.shape[0])
        cam = cv2.resize(cam, None, fx=s, fy=s)
        left = np.zeros((half[1], half[0], 3), np.uint8)
        y, x = (half[1] - cam.shape[0]) // 2, (half[0] - cam.shape[1]) // 2
        left[y : y + cam.shape[0], x : x + cam.shape[1]] = cam
        cx, cy = half[0] // 2, half[1] // 2  # camera centre = drone position
        cv2.drawMarker(left, (cx, cy), COLORS["route"], cv2.MARKER_CROSS, 30, 3)
        put_text(left, f"camera {f['label']}", (10, 30), 0.8, (255, 255, 255), 2)

        right, origin = crop_around(img, p, vw, vh)
        cv2.polylines(
            right, [np.int32(path[: i + 1] - origin)], False, COLORS["flown"], 4
        )
        cv2.polylines(
            right, [np.int32(f["corners"] - origin)], True, COLORS["footprint"], 4
        )
        cv2.circle(right, tuple(np.int32(p - origin)), 12, COLORS["route"], -1)
        right = cv2.resize(right, half)
        put_text(right, "position on map", (10, 30), 0.8, (255, 255, 255), 2)
        video.write(np.hstack([left, right]))
    video.release()


def save_csv(path, frames):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "label", "found", "x", "y", "inliers"])
        for f in frames:
            x, y = f["center"] if f["ok"] else ("", "")
            w.writerow(
                [
                    f["name"],
                    f["label"],
                    int(f["ok"]),
                    f"{x:.1f}" if f["ok"] else x,
                    f"{y:.1f}" if f["ok"] else y,
                    f["inliers"],
                ]
            )


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
        frames = frames_from_video(
            args.from_video, args.every, os.path.join(out_dir, "video_frames")
        )
    else:
        frames = frames_from_folder(args.frames)

    # 2. Localise every frame on the map, then remove obviously wrong positions
    locate_frames(map_gray, frames, screen_video)
    drop_wrong_sizes(frames)
    drop_outliers(frames)
    found = [f for f in frames if f["ok"]]
    print(f"Located {len(found)} of {len(frames)} frames")
    if len(found) < 2:
        raise SystemExit("Too few frames located to draw a route")

    # 3. Points keep their own labels even if some frames were skipped.
    #    Frame positions from a folder are already accurate to a pixel, so they are drawn as
    #    found (smoothing would pull points off their footprints). A video gives many close
    #    points with more jitter, so it is smoothed (START and FINISH are never moved).
    path = np.array([f["center"] for f in found])
    if screen_video:
        path = smooth(path, window=5)

    # 4. Route image and CSV with positions
    number_text = "label = seconds into video" if screen_video else "number = frame"
    img = draw_route(
        map_img,
        found,
        path,
        number_text,
        max_footprints=MAX_VIDEO_FOOTPRINTS if screen_video else None,
    )
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
