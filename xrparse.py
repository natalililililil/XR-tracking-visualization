"""Parser for trackingData_*.txt (XRoboToolkit / PICO headset recording).

Format notes (found by inspecting the data):
* Line 1 is a header: timeStampNs of the first video frame, camera extrinsics/intrinsics.
* Every following line is one JSON object (~90 Hz); time is timeStampNs (Unix epoch, ns).
* Numbers are written with a DECIMAL COMMA (the recorder serialized floats in a Russian locale)
  and are also separated by commas. So "-0,085,-0,027,..." is NOT 7 values split by commas:
  it is 14 tokens = 7 numbers, each written as (integer part, fractional part).
"""
import json
import numpy as np

# Hand joint order follows OpenXR XR_EXT_hand_tracking (26 joints per hand)
JOINT_NAMES = ["Palm", "Wrist",
               "ThumbMeta", "ThumbProx", "ThumbDist", "ThumbTip",
               "IndexMeta", "IndexProx", "IndexInter", "IndexDist", "IndexTip",
               "MiddleMeta", "MiddleProx", "MiddleInter", "MiddleDist", "MiddleTip",
               "RingMeta", "RingProx", "RingInter", "RingDist", "RingTip",
               "LittleMeta", "LittleProx", "LittleInter", "LittleDist", "LittleTip"]
# Joint indices forming each finger (used to draw a hand skeleton)
FINGERS = {"thumb": [1, 2, 3, 4, 5], "index": [1, 6, 7, 8, 9, 10], "middle": [1, 11, 12, 13, 14, 15],
           "ring": [1, 16, 17, 18, 19, 20], "little": [1, 21, 22, 23, 24, 25]}


def parse_nums(s, n):
    """'-0,08,-0,02,0,16,...' -> list of n floats, or None if the token count is not 2*n.

    A value printed without a fractional part (e.g. an exact '0') would make pairing
    ambiguous, so such a string is rejected instead of being silently mis-decoded.
    """
    t = s.split(",")
    if len(t) != 2 * n:
        return None
    try:
        return [float(f"{t[2*i]}.{t[2*i+1]}") for i in range(n)]
    except ValueError:
        return None


def load_header(path):
    """First line: video start time and camera calibration (these use normal dots)."""
    with open(path, encoding="utf-8") as f:
        h = json.loads(f.readline())
    ext = [json.loads(x) for x in h["cameraExtrinsics"].split("|")]  # left eye first, then right
    return {"t0_ns": h["timeStampNs"],
            "extrinsics": [np.array(e).reshape(4, 4) for e in ext],
            "intrinsics": json.loads(h["cameraIntrinsics"])}


def load_tracking(path):
    """Return a dict of arrays:
    t        (N,)        seconds since the first video frame
    head     (N, 7)      x, y, z, qx, qy, qz, qw
    hand_l/r (N, 26, 7)  joint poses; NaN while the hand is not tracked
    act_l/r  (N,)        1 if the hand is tracked (isActive == 1), else 0
    header               calibration from the first line
    """
    hdr = load_header(path)
    t, head, hl, hr, al, ar = [], [], [], [], [], []
    bad = 0
    with open(path, encoding="utf-8") as f:
        next(f)  # skip the header line
        for line in f:
            try:
                d = json.loads(line)
                h = parse_nums(d["Head"]["pose"], 7)
                if h is None:
                    raise ValueError
                row = {}
                for side in ("left", "right"):
                    hd = d["Hand"][side + "Hand"]
                    j = [parse_nums(x["p"], 7) for x in hd["HandJointLocations"]]
                    # an inactive hand keeps stale coordinates in the log -> replace with NaN
                    ok = hd["isActive"] == 1 and all(x is not None for x in j)
                    row[side] = (np.array(j) if ok else np.full((26, 7), np.nan), int(ok))
                t.append((d["timeStampNs"] - hdr["t0_ns"]) / 1e9)
                head.append(h)
                hl.append(row["left"][0]); al.append(row["left"][1])
                hr.append(row["right"][0]); ar.append(row["right"][1])
            except Exception:
                bad += 1
    print(f"tracking samples: {len(t)}, skipped lines: {bad}")
    return dict(t=np.array(t), head=np.array(head), hand_l=np.array(hl), hand_r=np.array(hr),
                act_l=np.array(al), act_r=np.array(ar), header=hdr)