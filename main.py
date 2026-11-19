import argparse, glob, re
import cv2, numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

def read(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, np.uint8), flag)

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True)
ap.add_argument("--frames", required=True)
ap.add_argument("--out", default="trajectory.png")
ap.add_argument("--video", default="trajectory.mp4")
args = ap.parse_args()

# 1. Map (downscaled if huge) and its SIFT features
map_img = read(args.map)
s = min(1, 4000 / max(map_img.shape[:2]))
map_img = cv2.resize(map_img, None, fx=s, fy=s)
sift = cv2.SIFT_create(nfeatures=20000)
kp_m, des_m = sift.detectAndCompute(cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY), None)
bf = cv2.BFMatcher()

# 2. Localise every frame (in natural order: crop_2 before crop_10)
files = sorted(glob.glob(args.frames + "/*"),
               key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", f)])
pts, boxes = [], []
for f in files:
    img = read(f, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    if des is None:
        continue
    good = [m for m, n in bf.knnMatch(des, des_m, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        print(f, "- too few matches, skipped")
        continue
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_m[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        continue
    h, w = img.shape
    # 4 corners of the frame + its centre -> map coordinates
    pr = cv2.perspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]]).reshape(-1, 1, 2), H).reshape(-1, 2)
    boxes.append(pr[:4])
    pts.append(pr[4])
    print(f, "- ok")

# 3. Smooth the trajectory (moving average over 3 points)
pts = np.array(pts)
pad = np.pad(pts, ((1, 1), (0, 0)), "edge")
path = np.stack([np.convolve(pad[:, i], np.ones(3) / 3, "valid") for i in range(2)], 1)

# 4. Draw: frame footprints, route, numbered points, START / FINISH
img = map_img.copy()
cv2.polylines(img, [np.int32(b) for b in boxes], True, (255, 200, 0), 2)
cv2.polylines(img, [np.int32(path)], False, (0, 0, 255), 4)
for i, p in enumerate(path):
    x, y = map(int, p)
    cv2.circle(img, (x, y), 8, (0, 255, 0), -1)
    cv2.putText(img, str(i + 1), (x + 10, y - 10), FONT, 1, (255, 255, 255), 2)
for p, label, color in [(path[0], "START", (255, 0, 0)), (path[-1], "FINISH", (0, 255, 255))]:
    x, y = map(int, p)
    cv2.circle(img, (x, y), 16, color, 3)
    cv2.putText(img, label, (x + 20, y + 30), FONT, 1.2, color, 3)
cv2.imencode(".png", img)[1].tofile(args.out)

# 5. Video: zoomed map region follows the UAV
map_h, map_w = img.shape[:2]
vw_, vh_ = 800, 450
video = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1280, 720))
for a, b in zip(path[:-1], path[1:]):
    for t in np.linspace(0, 1, 10, endpoint=False):      # interpolation between points
        p = a + (b - a) * t
        x0 = int(np.clip(p[0] - vw_ / 2, 0, map_w - vw_))
        y0 = int(np.clip(p[1] - vh_ / 2, 0, map_h - vh_))
        fr = img[y0:y0 + vh_, x0:x0 + vw_].copy()
        cv2.circle(fr, (int(p[0] - x0), int(p[1] - y0)), 10, (0, 0, 255), -1)
        video.write(cv2.resize(fr, (1280, 720)))
video.release()
print("Done:", args.out, args.video)