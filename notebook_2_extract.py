import os
import cv2
import json
import warnings
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")
print("Libraries ready")

# ── PATHS ──────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw_videos")
OUTPUT_DIR = Path("data/processed")
MODEL_DIR  = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── AUTO DETECT ASANAS ─────────────────────────────────────────
ASANAS     = [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]
LABEL_MAP  = {"poor": 0, "avg": 1, "good": 2}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}

print("Data folder :", DATA_DIR.resolve())
print("Asanas found:", ASANAS)

# ── 35 FEATURE NAMES ───────────────────────────────────────────
FEATURE_NAMES = [
    "left_elbow",       "right_elbow",
    "left_shoulder",    "right_shoulder",
    "left_arm_raise",   "right_arm_raise",
    "left_wrist_bend",  "right_wrist_bend",
    "left_hip",         "right_hip",
    "left_knee",        "right_knee",
    "left_ankle",       "right_ankle",
    "left_foot",        "right_foot",
    "left_hip_abduction", "right_hip_abduction",
    "neck_tilt",        "spine_upper",
    "spine_lower",      "left_trunk",
    "right_trunk",      "left_lateral_bend",
    "forward_bend",     "pelvic_tilt",
    "left_arm_torso",   "right_arm_torso",
    "left_shoulder_plane", "right_shoulder_plane",
    "left_hip_flexion", "right_hip_flexion",
    "hip_alignment",
    "left_full_chain",  "right_full_chain",
]

print("Total features:", len(FEATURE_NAMES))

# ── ANGLE FUNCTIONS ────────────────────────────────────────────
def angle_between(a, b, c):
    ba  = a - b
    bc  = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def pt(kp, i):
    return kp[i, :3]

def mid(kp, i, j):
    return (kp[i, :3] + kp[j, :3]) / 2

def compute_joint_angles(kp):
    critical = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
    if np.any(kp[critical, 3] < 0.5):
        return None
    try:
        mid_sh    = mid(kp, 11, 12)
        mid_hip   = mid(kp, 23, 24)
        mid_knee  = mid(kp, 25, 26)
        mid_ankle = mid(kp, 27, 28)
        return np.array([
            # Arms
            angle_between(pt(kp,11), pt(kp,13), pt(kp,15)),
            angle_between(pt(kp,12), pt(kp,14), pt(kp,16)),
            angle_between(pt(kp,13), pt(kp,11), pt(kp,23)),
            angle_between(pt(kp,14), pt(kp,12), pt(kp,24)),
            angle_between(pt(kp,23), pt(kp,11), pt(kp,13)),
            angle_between(pt(kp,24), pt(kp,12), pt(kp,14)),
            angle_between(pt(kp,13), pt(kp,15), pt(kp,19)),
            angle_between(pt(kp,14), pt(kp,16), pt(kp,20)),
            # Legs
            angle_between(pt(kp,11), pt(kp,23), pt(kp,25)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,26)),
            angle_between(pt(kp,23), pt(kp,25), pt(kp,27)),
            angle_between(pt(kp,24), pt(kp,26), pt(kp,28)),
            angle_between(pt(kp,25), pt(kp,27), pt(kp,29)),
            angle_between(pt(kp,26), pt(kp,28), pt(kp,30)),
            angle_between(pt(kp,27), pt(kp,29), pt(kp,31)),
            angle_between(pt(kp,28), pt(kp,30), pt(kp,32)),
            angle_between(pt(kp,26), pt(kp,24), pt(kp,23)),
            angle_between(pt(kp,25), pt(kp,23), pt(kp,24)),
            # Spine & Trunk
            angle_between(pt(kp,0),  mid_sh,    mid_hip),
            angle_between(mid_sh,    mid_hip,   mid_knee),
            angle_between(mid_hip,   mid_knee,  mid_ankle),
            angle_between(pt(kp,11), pt(kp,23), pt(kp,25)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,26)),
            angle_between(pt(kp,11), mid_sh,    pt(kp,12)),
            angle_between(pt(kp,0),  mid_sh,    mid_hip),
            angle_between(mid_sh,    mid_hip,   mid_ankle),
            # Shoulder complex
            angle_between(pt(kp,12), pt(kp,11), pt(kp,13)),
            angle_between(pt(kp,11), pt(kp,12), pt(kp,14)),
            angle_between(pt(kp,13), pt(kp,11), pt(kp,23)),
            angle_between(pt(kp,14), pt(kp,12), pt(kp,24)),
            # Hip complex
            angle_between(pt(kp,11), pt(kp,23), pt(kp,25)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,26)),
            angle_between(pt(kp,23), mid_hip,   pt(kp,24)),
            # Full chain
            angle_between(pt(kp,11), pt(kp,23), pt(kp,27)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,28)),
        ], dtype=np.float32)
    except:
        return None

# ── COLLECT ALL VIDEO PATHS ────────────────────────────────────
video_list = []
for asana_dir in sorted(DATA_DIR.iterdir()):
    if not asana_dir.is_dir():
        continue
    for label in ["good", "avg", "poor"]:
        label_dir = asana_dir / label
        if not label_dir.exists():
            print("Missing folder:", asana_dir.name + "/" + label)
            continue
        for vid in sorted(label_dir.iterdir()):
            if vid.suffix in VIDEO_EXTS:
                video_list.append({
                    "asana":     asana_dir.name,
                    "label":     label,
                    "label_int": LABEL_MAP[label],
                    "path":      str(vid),
                    "filename":  vid.name,
                })

print("Total videos:", len(video_list))

# ── MAIN EXTRACTION LOOP ───────────────────────────────────────
SAMPLE_FPS = 5
all_rows   = []
stats      = {"extracted": 0, "no_pose": 0, "low_vis": 0, "bad_video": 0}

mp_pose = mp.solutions.pose

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    for video in tqdm(video_list, desc="Processing videos"):
        cap = cv2.VideoCapture(video["path"])

        if not cap.isOpened():
            stats["poor_video"] += 1
            continue

        fps_v    = cap.get(cv2.CAP_PROP_FPS) or 25
        interval = max(1, int(fps_v / SAMPLE_FPS))
        idx      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = pose.process(rgb)

                if not result.pose_landmarks:
                    stats["no_pose"] += 1
                else:
                    kp = np.array([
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in result.pose_landmarks.landmark
                    ], dtype=np.float32)

                    angles = compute_joint_angles(kp)

                    if angles is None:
                        stats["low_vis"] += 1
                    else:
                        row = {
                            "asana":     video["asana"],
                            "label":     video["label"],
                            "label_int": video["label_int"],
                            "filename":  video["filename"],
                        }
                        for name, val in zip(FEATURE_NAMES, angles):
                            row[name] = round(float(val), 3)
                        all_rows.append(row)
                        stats["extracted"] += 1

            idx += 1
        cap.release()

# ── SAVE ───────────────────────────────────────────────────────
print("\nSaving...")
df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_DIR / "master_dataset.csv", index=False)

per_asana_dir = OUTPUT_DIR / "per_asana"
per_asana_dir.mkdir(exist_ok=True)
for asana, grp in df.groupby("asana"):
    grp.to_csv(per_asana_dir / (asana + ".csv"), index=False)

meta = {
    "feature_names": FEATURE_NAMES,
    "label_map":     LABEL_MAP,
    "asanas":        sorted(df["asana"].unique().tolist()),
    "total_frames":  len(df),
    "sample_fps":    SAMPLE_FPS,
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# ── SUMMARY ────────────────────────────────────────────────────
print("=" * 50)
print("EXTRACTION COMPLETE")
print("=" * 50)
print("Features per frame :", len(FEATURE_NAMES))
print("Frames extracted   :", stats["extracted"])
print("No pose detected   :", stats["no_pose"])
print("Low visibility     :", stats["low_vis"])
print("Bad videos         :", stats["bad_video"])
print("CSV rows saved     :", len(df))
print("Saved in           :", OUTPUT_DIR.resolve())
print("=" * 50)
print(df.groupby(["asana", "label"]).size().unstack(fill_value=0))
print("\nDone! Run notebook_3_train.py next.")
