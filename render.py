"""Video with a schematic tracking view (top + side) underneath, synchronized by timestamps.

Usage: python render.py [tracking.txt] [video.mp4] [out.mp4]
"""
import os, sys
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xrparse import load_tracking, FINGERS

TRACK = sys.argv[1] if len(sys.argv) > 1 else "trackingData_20260505_165740.txt"
VIDEO = sys.argv[2] if len(sys.argv) > 2 else "CameraRecord_20260505_165740.mp4"
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/side_by_side.mp4"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
TRAIL = 1.5  # seconds of trajectory "tail"

D = load_tracking(TRACK)
t = D["t"]
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS); W = int(cap.get(3)); H = int(cap.get(4))
print(f"video {W}x{H} @ {fps:.1f} fps, {cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps:.1f} s; tracking {t[0]:.2f}..{t[-1]:.2f} s")

# fixed axis limits from the whole recording so the view does not jump
allp = np.vstack([D["head"][:, :3], D["hand_l"][:, :, :3].reshape(-1, 3), D["hand_r"][:, :, :3].reshape(-1, 3)])
lo, hi = np.nanpercentile(allp, 1, 0) - .1, np.nanpercentile(allp, 99, 0) + .1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W / 100, W / 100 * .5), dpi=100)
a1.set(xlim=(lo[0], hi[0]), ylim=(lo[2], hi[2]), xlabel="X", ylabel="Z", title="Top view", aspect="equal")
a2.set(xlim=(lo[2], hi[2]), ylim=(lo[1], hi[1]), xlabel="Z", ylabel="Y (up)", title="Side view", aspect="equal")
for a in (a1, a2):
    a.grid(alpha=.3)

L = {}  # matplotlib artists, updated every frame (much faster than redrawing the figure)


def mk(name, col):
    L[name] = dict(trail=[a1.plot([], [], c=col, alpha=.4)[0], a2.plot([], [], c=col, alpha=.4)[0]],
                   dot=[a1.plot([], [], "o", c=col, ms=7)[0], a2.plot([], [], "o", c=col, ms=7)[0]],
                   bones=[[a.plot([], [], "-", c=col, lw=1)[0] for _ in range(5)] for a in (a1, a2)])


mk("head", "k"); mk("left", "tab:blue"); mk("right", "tab:red")
fig.tight_layout()


def draw(ts):
    """Render the tracking panel for video time ts and return it as a BGR image."""
    i = min(np.searchsorted(t, ts), len(t) - 1); j = np.searchsorted(t, ts - TRAIL)
    for name, P, act in (("head", D["head"][:, None, :3], True),
                         ("left", D["hand_l"][:, :, :3], D["act_l"][i]),
                         ("right", D["hand_r"][:, :, :3], D["act_r"][i])):
        c = P[:, 0] if name == "head" else P[:, 1]          # head, or the wrist joint
        for k, (ix, iy) in enumerate(((0, 2), (2, 1))):      # top view: (X, Z); side view: (Z, Y)
            L[name]["trail"][k].set_data(c[j:i + 1, ix], c[j:i + 1, iy])
            L[name]["dot"][k].set_data([c[i, ix]], [c[i, iy]] if act else [np.nan])  # hidden if not tracked
            if name != "head":
                for b, idx in enumerate(FINGERS.values()):
                    q = P[i, idx] if act else np.full((len(idx), 3), np.nan)
                    L[name]["bones"][k][b].set_data(q[:, ix], q[:, iy])
    fig.canvas.draw()
    return cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)


ph = fig.canvas.get_width_height()[1]
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + ph))
n = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    out.write(np.vstack([fr, draw(n / fps)])); n += 1  # video frame n <-> time n / fps after the start
out.release(); print("done:", OUT)