"""Static plots and statistics -> results/overview.png, results/trajectory3d.png.

Usage: python analyze.py [trackingData_*.txt]
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from xrparse import load_tracking

path = sys.argv[1] if len(sys.argv) > 1 else "trackingData_20260505_165740.txt"
os.makedirs("results", exist_ok=True)
D = load_tracking(path)
t, head = D["t"], D["head"]
WRIST = 1  # index of the Wrist joint in the OpenXR joint order


def speed(p):
    """Speed in m/s; NaN where the hand is not tracked."""
    dt = np.gradient(t)
    return np.linalg.norm(np.gradient(p, axis=0), axis=1) / np.maximum(dt, 1e-3)


wl, wr = D["hand_l"][:, WRIST, :3], D["hand_r"][:, WRIST, :3]
# head orientation as yaw / pitch / roll (approximate, Y-X-Z order), unwrapped to avoid 360° jumps
ypr = R.from_quat(head[:, 3:7]).as_euler("YXZ", degrees=True)
ypr = np.unwrap(np.radians(ypr), axis=0) * 180 / np.pi

fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
for i, (n, c) in enumerate(zip("XYZ", ("tab:red", "tab:green", "tab:blue"))):
    ax[0].plot(t, head[:, i], c=c, label=f"Head {n}")
ax[0].set(ylabel="m", title="Head position (relative to the tracking origin)"); ax[0].legend(ncol=3)
for i, n in enumerate(("yaw", "pitch", "roll")):
    ax[1].plot(t, ypr[:, i], label=n)
ax[1].set(ylabel="degrees", title="Head orientation"); ax[1].legend(ncol=3)
ax[2].plot(t, speed(wl), label="left wrist", c="tab:blue", alpha=.8)
ax[2].plot(t, speed(wr), label="right wrist", c="tab:red", alpha=.8)
ax[2].set(ylabel="m/s", title="Wrist speed (gaps = hand not tracked)"); ax[2].legend()
ax[3].fill_between(t, 0, D["act_l"], step="post", alpha=.5, label="left hand tracked", color="tab:blue")
ax[3].fill_between(t, 1.1, 1.1 + D["act_r"], step="post", alpha=.5, label="right hand tracked", color="tab:red")
ax[3].set(xlabel="time since video start, s", yticks=[], title="When each hand is tracked (isActive)")
ax[3].legend(loc="center right")
for a in ax:
    a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/overview.png", dpi=150)

# 3D trajectories (Y is up, Z инвертирован для зеркального отображения лево/право)
fig = plt.figure(figsize=(8, 7)); a = fig.add_subplot(111, projection="3d")
for p, n, c in ((head[:, :3], "head", "k"), (wl, "left wrist", "tab:blue"), (wr, "right wrist", "tab:red")):
    a.plot(p[:, 0], -p[:, 2], p[:, 1], c=c, label=n)

a.set(xlabel="X", ylabel="-Z", zlabel="Y (up)"); a.legend(); a.set_title("3D trajectories")

fig.savefig("results/trajectory3d.png", dpi=150)

# ---- console statistics ----
fps = 1 / np.median(np.diff(t))
print(f"duration {t[-1]-t[0]:.1f} s, ~{fps:.0f} Hz, dt min {np.diff(t).min()*1e3:.1f} / max {np.diff(t).max()*1e3:.1f} ms")


def segments(act):
    """Number and durations of continuous tracked intervals."""
    a = np.concatenate([[0], act.astype(int), [0]]); d = np.diff(a)
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    return len(st), (t[np.minimum(en, len(t) - 1)] - t[st]) if len(st) else np.array([])


for name, act in (("left", D["act_l"]), ("right", D["act_r"])):
    n, dur = segments(act)
    print(f"{name} hand: tracked {act.sum() / len(act) * 100:.0f}% of the time, {n} segments" +
          (f", longest {dur.max():.1f} s" if n else ""))
print(f"head path length {np.nansum(np.linalg.norm(np.diff(head[:, :3], axis=0), axis=1)):.2f} m")
print(f"max wrist speed: left {np.nanmax(speed(wl)):.2f} m/s, right {np.nanmax(speed(wr)):.2f} m/s")