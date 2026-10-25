"""Project tracked hands from trackingData_*.txt onto the stereo video CameraRecord_*.mp4.

Needs xrparse.py, the video and the tracking file.
  python overlay.py --t 9               # ONE frame  -> results/overlay_9.0s.png
  python overlay.py --start 5 --dur 20  # a clip     -> results/overlay.mp4
  python overlay.py                     # full video -> results/overlay.mp4
  python overlay.py --tune --t 9        # window with sliders to find an empirical offset

Pipeline: world joint -> head frame (from Head.pose) -> camera frame (cameraExtrinsics)
-> pixels (cameraIntrinsics).
Assumptions: OpenXR axes (x right, y up, -z forward); extrinsics are given in device axes
(x up, y left, z back), which needs a fixed axis permutation; the video is a stereo pair
2 x (1080x810) where each camera image is squeezed horizontally by 2, so
fx_video = fx / 2 and cx_video = cx / 2. Hands that are not tracked (isActive = 0) are not drawn.
"""
import argparse, os
import numpy as np, cv2
from scipy.spatial.transform import Rotation as R
from xrparse import load_tracking, FINGERS

ap = argparse.ArgumentParser()
ap.add_argument("--track", default="trackingData_20260505_165740.txt")
ap.add_argument("--video", default="CameraRecord_20260505_165740.mp4")
ap.add_argument("--t", type=float, help="time of the frame in seconds (single frame / --tune mode)")
ap.add_argument("--start", type=float, default=0); ap.add_argument("--dur", type=float)
ap.add_argument("--offset", type=float, default=0.0, help="tracking time shift, s")
ap.add_argument("--shift", type=float, nargs=3, default=[0, 0, 0], metavar=("DX", "DY", "DZ"),
                help="empirical hand shift in the head frame, m (x right, y up, z back)")
ap.add_argument("--eyes", type=int, choices=[1, 2], default=2, help="1 = left camera only")
ap.add_argument("--labels", action="store_true", help="print coordinates from the log on the frame")
ap.add_argument("--tune", action="store_true")
a = ap.parse_args()
os.makedirs("results", exist_ok=True)

D = load_tracking(a.track); t = D["t"]; hdr = D["header"]
cx, cy, fx2, fy = hdr["intrinsics"]                  # header stores [cx, cy, fx, fy] for the unsqueezed 2160x810 image
K = dict(fx=fx2 / 2, fy=fy, cx=cx / 2, cy=cy)        # parameters of ONE camera in the video (1080x810)
M = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])    # device axes -> OpenXR axes
CAMS = [(M @ E[:3, :3], M @ E[:3, 3]) for E in hdr["extrinsics"][:2]]  # camera -> head (left, right)
P = dict(offset=a.offset, shift=np.array(a.shift, float))
COL = {"left": (255, 120, 0), "right": (0, 0, 255)}  # BGR


def project(Pw, head, cam_i, shift, eye_w=1080):
    """World points (N,3) -> pixels (N,2) in the full side-by-side frame; NaN if behind the camera."""
    p_head = R.from_quat(head[3:7]).inv().apply(Pw - head[:3]) + shift
    Rc, tc = CAMS[cam_i]
    pc = (p_head - tc) @ Rc                          # = Rc^T (p - tc): camera coordinates (OpenCV axes)
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = np.stack([K["fx"] * pc[:, 0] / pc[:, 2] + K["cx"], K["fy"] * pc[:, 1] / pc[:, 2] + K["cy"]], 1)
    uv[pc[:, 2] <= 0.05] = np.nan                    # too close to / behind the camera
    return uv + np.array([cam_i * eye_w, 0])         # right camera lives in the right half of the frame


def draw(frame, ts):
    """Draw hand skeletons for video time ts on the frame."""
    i = int(min(np.searchsorted(t, ts - P["offset"]), len(t) - 1))
    inside = lambda p: np.isfinite(p).all() and -50 < p[0] < 2210 and -50 < p[1] < 860
    txt = [f"t={ts:.2f}s  idx={i}  head=({D['head'][i,0]:+.2f},{D['head'][i,1]:+.2f},{D['head'][i,2]:+.2f})"]
    for side, J, act in (("left", D["hand_l"][i], D["act_l"][i]), ("right", D["hand_r"][i], D["act_r"][i])):
        if not act:                                  # hand not tracked -> nothing is drawn
            txt.append(f"{side}: not tracked"); continue
        w = J[1, :3]; txt.append(f"{side} wrist=({w[0]:+.2f},{w[1]:+.2f},{w[2]:+.2f})")
        for cam_i in range(a.eyes):
            uv = project(J[:, :3], D["head"][i], cam_i, P["shift"])
            for idx in FINGERS.values():
                for u, v in zip(idx[:-1], idx[1:]):
                    if inside(uv[u]) and inside(uv[v]):
                        cv2.line(frame, tuple(map(int, uv[u])), tuple(map(int, uv[v])), COL[side], 2, cv2.LINE_AA)
            for p in uv:
                if inside(p):
                    cv2.circle(frame, tuple(map(int, p)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if not a.labels:
        txt = txt[:1]
    for k, s in enumerate(txt):
        cv2.putText(frame, s, (20, 40 + 36 * k), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)


def read_at(sec):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(sec * fps))); ok, fr = cap.read(); return fr if ok else None


if a.tune:
    # Interactive: sliders for a hand shift (mm) and a time shift (ms) on one frame; 'q' prints the command.
    t0 = a.t if a.t is not None else 9.0
    base = read_at(t0); win = "tune (q = quit)"; cv2.namedWindow(win)
    for name, init, rng in (("dx mm", 200, 400), ("dy mm", 200, 400), ("dz mm", 200, 400), ("time ms", 300, 600)):
        cv2.createTrackbar(name, win, init, rng, lambda v: None)
    while True:
        P["shift"] = np.array([cv2.getTrackbarPos(n, win) - 200 for n in ("dx mm", "dy mm", "dz mm")]) / 1000
        P["offset"] = (cv2.getTrackbarPos("time ms", win) - 300) / 1000
        cv2.imshow(win, cv2.resize(draw(base.copy(), t0), None, fx=.6, fy=.6))
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    s = P["shift"]; print(f"python overlay.py --shift {s[0]:.3f} {s[1]:.3f} {s[2]:.3f} --offset {P['offset']:.3f}")
elif a.t is not None:                                # one frame -> PNG
    out = f"results/overlay_{a.t:.1f}s.png"; cv2.imwrite(out, draw(read_at(a.t), a.t)); print("saved", out)
else:                                                # video (whole or a clip)
    n0 = int(a.start * fps); cap.set(cv2.CAP_PROP_POS_FRAMES, n0)
    wr = cv2.VideoWriter("results/overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(cap.get(3)), int(cap.get(4))))
    for k in range(int(a.dur * fps) if a.dur else 10**9):
        ok, fr = cap.read()
        if not ok:
            break
        wr.write(draw(fr, (n0 + k) / fps))
    wr.release(); print("done: results/overlay.mp4")