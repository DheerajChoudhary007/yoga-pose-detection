import cv2
import json
import joblib
import warnings
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
from collections import deque

warnings.filterwarnings("ignore")

# ── PATHS ──────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/processed")
MODEL_DIR  = Path("models")

# ── LOAD METADATA ──────────────────────────────────────────────
with open(OUTPUT_DIR / "metadata.json") as f:
    meta = json.load(f)

FEATURE_NAMES = meta["feature_names"]
LABEL_MAP     = meta["label_map"]
LABEL_NAMES   = {v: k for k, v in LABEL_MAP.items()}
ASANAS        = meta["asanas"]

# ── LOAD ALL MODELS ────────────────────────────────────────────
models = {}
for asana in ASANAS:
    p = MODEL_DIR / (asana + "_model.pkl")
    if p.exists():
        models[asana] = joblib.load(p)
        print("Loaded model:", asana)

# ── LOAD REFERENCE ANGLES ──────────────────────────────────────
df_all    = pd.read_csv(OUTPUT_DIR / "master_dataset.csv")
reference = {}
for asana in ASANAS:
    good = df_all[(df_all["asana"] == asana) & (df_all["label"] == "good")]
    if len(good) == 0:
        continue
    reference[asana] = {
        feat: {"mean": float(good[feat].mean()), "std": float(good[feat].std())}
        for feat in FEATURE_NAMES if feat in good.columns
    }

print("Reference angles loaded")
print("Available asanas:", ASANAS)

# ── SETTINGS ───────────────────────────────────────────────────
# 👇 Change this to the asana you want to practice
CURRENT_ASANA       = "tadasana"
DEVIATION_THRESHOLD = 15.0
SMOOTHING_FRAMES    = 10
MAX_FEEDBACK        = 4

# ── FEEDBACK MESSAGES ──────────────────────────────────────────
FEEDBACK = {
    "left_elbow":           {"too_low": "Straighten LEFT ELBOW",          "too_high": "Bend LEFT ELBOW more"},
    "right_elbow":          {"too_low": "Straighten RIGHT ELBOW",         "too_high": "Bend RIGHT ELBOW more"},
    "left_shoulder":        {"too_low": "Raise LEFT SHOULDER",            "too_high": "Lower LEFT SHOULDER"},
    "right_shoulder":       {"too_low": "Raise RIGHT SHOULDER",           "too_high": "Lower RIGHT SHOULDER"},
    "left_arm_raise":       {"too_low": "Raise LEFT ARM higher",          "too_high": "Lower LEFT ARM slightly"},
    "right_arm_raise":      {"too_low": "Raise RIGHT ARM higher",         "too_high": "Lower RIGHT ARM slightly"},
    "left_wrist_bend":      {"too_low": "Straighten LEFT WRIST",          "too_high": "Flex LEFT WRIST more"},
    "right_wrist_bend":     {"too_low": "Straighten RIGHT WRIST",         "too_high": "Flex RIGHT WRIST more"},
    "left_hip":             {"too_low": "Open LEFT HIP more",             "too_high": "Close LEFT HIP slightly"},
    "right_hip":            {"too_low": "Open RIGHT HIP more",            "too_high": "Close RIGHT HIP slightly"},
    "left_knee":            {"too_low": "Straighten LEFT KNEE",           "too_high": "Bend LEFT KNEE more"},
    "right_knee":           {"too_low": "Straighten RIGHT KNEE",          "too_high": "Bend RIGHT KNEE more"},
    "left_ankle":           {"too_low": "Flex LEFT ANKLE upward",         "too_high": "Point LEFT ANKLE down"},
    "right_ankle":          {"too_low": "Flex RIGHT ANKLE upward",        "too_high": "Point RIGHT ANKLE down"},
    "left_foot":            {"too_low": "Press LEFT FOOT down",           "too_high": "Lift LEFT FOOT slightly"},
    "right_foot":           {"too_low": "Press RIGHT FOOT down",          "too_high": "Lift RIGHT FOOT slightly"},
    "left_hip_abduction":   {"too_low": "Spread LEFT LEG outward",        "too_high": "Bring LEFT LEG inward"},
    "right_hip_abduction":  {"too_low": "Spread RIGHT LEG outward",       "too_high": "Bring RIGHT LEG inward"},
    "neck_tilt":            {"too_low": "Lift your HEAD up",              "too_high": "Lower your CHIN slightly"},
    "spine_upper":          {"too_low": "Straighten UPPER SPINE",         "too_high": "Lengthen UPPER SPINE"},
    "spine_lower":          {"too_low": "Straighten LOWER SPINE",         "too_high": "Lengthen LOWER SPINE"},
    "left_trunk":           {"too_low": "Open LEFT side body",            "too_high": "Reduce LEFT lean"},
    "right_trunk":          {"too_low": "Open RIGHT side body",           "too_high": "Reduce RIGHT lean"},
    "left_lateral_bend":    {"too_low": "Bend LEFT laterally more",       "too_high": "Reduce LEFT lateral bend"},
    "forward_bend":         {"too_low": "Lean FORWARD more",              "too_high": "Stand more UPRIGHT"},
    "pelvic_tilt":          {"too_low": "Tuck PELVIS in",                 "too_high": "Release PELVIC tuck"},
    "left_arm_torso":       {"too_low": "Raise LEFT ARM from body",       "too_high": "Bring LEFT ARM closer"},
    "right_arm_torso":      {"too_low": "Raise RIGHT ARM from body",      "too_high": "Bring RIGHT ARM closer"},
    "left_shoulder_plane":  {"too_low": "Rotate LEFT SHOULDER back",      "too_high": "Bring LEFT SHOULDER forward"},
    "right_shoulder_plane": {"too_low": "Rotate RIGHT SHOULDER back",     "too_high": "Bring RIGHT SHOULDER forward"},
    "left_hip_flexion":     {"too_low": "Flex LEFT HIP more",             "too_high": "Extend LEFT HIP more"},
    "right_hip_flexion":    {"too_low": "Flex RIGHT HIP more",            "too_high": "Extend RIGHT HIP more"},
    "hip_alignment":        {"too_low": "Level your HIPS",                "too_high": "Square your HIPS"},
    "left_full_chain":      {"too_low": "Align LEFT side body",           "too_high": "Lengthen LEFT side"},
    "right_full_chain":     {"too_low": "Align RIGHT side body",          "too_high": "Lengthen RIGHT side"},
}

# ── ANGLE FUNCTIONS ────────────────────────────────────────────
def angle_between(a, b, c):
    ba  = a - b
    bc  = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def pt(kp, i): return kp[i, :3]
def mid(kp, i, j): return (kp[i, :3] + kp[j, :3]) / 2

def compute_joint_angles(kp):
    critical = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
    if np.any(kp[critical, 3] < 0.3):
        return None
    try:
        mid_sh    = mid(kp, 11, 12)
        mid_hip   = mid(kp, 23, 24)
        mid_knee  = mid(kp, 25, 26)
        mid_ankle = mid(kp, 27, 28)
        return np.array([
            angle_between(pt(kp,11), pt(kp,13), pt(kp,15)),
            angle_between(pt(kp,12), pt(kp,14), pt(kp,16)),
            angle_between(pt(kp,13), pt(kp,11), pt(kp,23)),
            angle_between(pt(kp,14), pt(kp,12), pt(kp,24)),
            angle_between(pt(kp,23), pt(kp,11), pt(kp,13)),
            angle_between(pt(kp,24), pt(kp,12), pt(kp,14)),
            angle_between(pt(kp,13), pt(kp,15), pt(kp,19)),
            angle_between(pt(kp,14), pt(kp,16), pt(kp,20)),
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
            angle_between(pt(kp,0),  mid_sh,    mid_hip),
            angle_between(mid_sh,    mid_hip,   mid_knee),
            angle_between(mid_hip,   mid_knee,  mid_ankle),
            angle_between(pt(kp,11), pt(kp,23), pt(kp,25)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,26)),
            angle_between(pt(kp,11), mid_sh,    pt(kp,12)),
            angle_between(pt(kp,0),  mid_sh,    mid_hip),
            angle_between(mid_sh,    mid_hip,   mid_ankle),
            angle_between(pt(kp,12), pt(kp,11), pt(kp,13)),
            angle_between(pt(kp,11), pt(kp,12), pt(kp,14)),
            angle_between(pt(kp,13), pt(kp,11), pt(kp,23)),
            angle_between(pt(kp,14), pt(kp,12), pt(kp,24)),
            angle_between(pt(kp,11), pt(kp,23), pt(kp,25)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,26)),
            angle_between(pt(kp,23), mid_hip,   pt(kp,24)),
            angle_between(pt(kp,11), pt(kp,23), pt(kp,27)),
            angle_between(pt(kp,12), pt(kp,24), pt(kp,28)),
        ], dtype=np.float32)
    except:
        return None

# ── JOINT COLOR MAPPING ────────────────────────────────────────
JOINT_TO_LANDMARKS = {
    "left_elbow": [13], "right_elbow": [14],
    "left_shoulder": [11], "right_shoulder": [12],
    "left_hip": [23], "right_hip": [24],
    "left_knee": [25], "right_knee": [26],
    "left_ankle": [27], "right_ankle": [28],
    "neck_tilt": [0], "spine_upper": [11, 12],
    "spine_lower": [23, 24], "left_trunk": [11, 23],
    "right_trunk": [12, 24], "hip_alignment": [23, 24],
    "left_full_chain": [11, 23, 27], "right_full_chain": [12, 24, 28],
    "left_arm_raise": [11, 13], "right_arm_raise": [12, 14],
    "left_wrist_bend": [15], "right_wrist_bend": [16],
    "left_foot": [29, 31], "right_foot": [30, 32],
}

# ── DRAW HELPERS ───────────────────────────────────────────────
def draw_box(frame, text, pos, color, scale=0.6, thick=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x, y = pos
    cv2.rectangle(frame, (x-5, y-th-5), (x+tw+5, y+5), (0,0,0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thick)

def get_joint_colors(joint_deviations):
    colors = {}
    for feat, info in joint_deviations.items():
        abs_dev = abs(info["deviation"])
        color   = (0,220,0) if abs_dev <= DEVIATION_THRESHOLD else \
                  (0,220,220) if abs_dev <= 25 else (0,0,220)
        for idx in JOINT_TO_LANDMARKS.get(feat, []):
            colors[idx] = color
    return colors

def draw_skeleton(frame, landmarks, w, h, joint_colors):
    connections = [
        (11,13),(13,15),(12,14),(14,16),
        (11,12),(11,23),(12,24),(23,24),
        (23,25),(25,27),(27,29),(27,31),
        (24,26),(26,28),(28,30),(28,32),
        (0,11),(0,12),
    ]
    for a, b in connections:
        la, lb = landmarks[a], landmarks[b]
        if la.visibility > 0.4 and lb.visibility > 0.4:
            x1,y1 = int(la.x*w), int(la.y*h)
            x2,y2 = int(lb.x*w), int(lb.y*h)
            cv2.line(frame, (x1,y1), (x2,y2), (180,180,180), 2)
    for i in range(33):
        lm = landmarks[i]
        if lm.visibility > 0.4:
            x, y  = int(lm.x*w), int(lm.y*h)
            color = joint_colors.get(i, (0,220,0))
            cv2.circle(frame, (x,y), 6, color, -1)
            cv2.circle(frame, (x,y), 8, (255,255,255), 1)

# ── MAIN REAL TIME LOOP ────────────────────────────────────────
def run_realtime(asana_name):
    if asana_name not in models:
        print("No model for:", asana_name)
        print("Available   :", list(models.keys()))
        return

    model       = models[asana_name]
    ref         = reference.get(asana_name, {})
    pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

    mp_pose_model = mp.solutions.pose
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    print("\nCamera started!")
    print("Practicing:", asana_name.upper())
    print("Press Q to quit.\n")

    with mp_pose_model.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,95), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            draw_box(frame, "Yoga: " + asana_name.upper() + "  |  Q = quit",
                     (10, 30), (255,255,255), 0.7, 2)

            # Legend
            cv2.circle(frame, (w-220, 65), 6, (0,220,0),   -1)
            draw_box(frame, "Good",    (w-210, 70), (0,220,0),   0.45, 1)
            cv2.circle(frame, (w-155, 65), 6, (0,220,220), -1)
            draw_box(frame, "Close",   (w-145, 70), (0,220,220), 0.45, 1)
            cv2.circle(frame, (w-90,  65), 6, (0,0,220),   -1)
            draw_box(frame, "Fix",     (w-80,  70), (0,0,220),   0.45, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                kp = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in landmarks
                ], dtype=np.float32)

                angles = compute_joint_angles(kp)

                if angles is not None:
                    pred_int = model.predict(angles.reshape(1,-1))[0]
                    pred_buffer.append(pred_int)
                    smooth   = max(set(pred_buffer), key=list(pred_buffer).count)
                    label    = LABEL_NAMES[smooth]

                    joint_deviations = {}
                    feedback_list    = []

                    for i, feat in enumerate(FEATURE_NAMES):
                        if feat not in ref:
                            continue
                        deviation = float(angles[i]) - ref[feat]["mean"]
                        abs_dev   = abs(deviation)
                        joint_deviations[feat] = {"deviation": deviation, "abs_dev": abs_dev}

                        if abs_dev > DEVIATION_THRESHOLD and feat in FEEDBACK:
                            direction = "too_low" if deviation < 0 else "too_high"
                            msg = FEEDBACK[feat].get(direction, "")
                            if msg:
                                feedback_list.append((abs_dev, msg))

                    feedback_list.sort(reverse=True)

                    joint_colors = get_joint_colors(joint_deviations)
                    draw_skeleton(frame, landmarks, w, h, joint_colors)

                    if label == "good":
                        badge_color, badge_text = (0,200,0),   "GOOD POSE"
                    elif label == "average":
                        badge_color, badge_text = (0,200,200), "AVERAGE POSE"
                    else:
                        badge_color, badge_text = (0,0,200),   "INCORRECT POSE"

                    draw_box(frame, badge_text, (10, 70), badge_color, 0.85, 2)

                    if feedback_list:
                        draw_box(frame, "Corrections:", (10, 115), (255,255,0), 0.6, 1)
                        for i, (dev, msg) in enumerate(feedback_list[:MAX_FEEDBACK]):
                            color  = (0,0,255) if dev > 25 else (0,200,255)
                            prefix = "[HIGH] " if dev > 25 else "[MED]  "
                            draw_box(frame, prefix + msg,
                                     (10, 143 + i*30), color, 0.57, 1)
                    else:
                        draw_box(frame, "Perfect! All joints correct!",
                                 (10, 115), (0,255,0), 0.7, 2)

                    # Score bar
                    good_count = sum(1 for p in pred_buffer if LABEL_NAMES[p]=="good")
                    score      = int(good_count / max(len(pred_buffer),1) * 100)
                    bar_w      = int((w-20) * score / 100)
                    cv2.rectangle(frame, (10,h-25), (w-10,h-10), (50,50,50), -1)
                    cv2.rectangle(frame, (10,h-25), (10+bar_w,h-10), (0,200,0), -1)
                    draw_box(frame, "Score: "+str(score)+"%",
                             (15,h-28), (255,255,255), 0.45, 1)

                    bad = sum(1 for f in joint_deviations.values()
                              if f["abs_dev"] > DEVIATION_THRESHOLD)
                    draw_box(frame,
                             "Joints to fix: "+str(bad)+"/"+str(len(FEATURE_NAMES)),
                             (10,h-48), (200,200,200), 0.48, 1)

                else:
                    draw_box(frame, "Show full body in frame",
                             (10, 70), (0,200,255), 0.7, 2)
            else:
                draw_box(frame, "No person detected — stand in front of camera",
                         (10, 70), (0,200,255), 0.65, 2)

            cv2.imshow("Yoga Pose Correction — " + asana_name.upper(), frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")


# ── RUN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Available asanas:", ASANAS)
    print("Currently set to:", CURRENT_ASANA)
    print("Edit CURRENT_ASANA at top of file to change\n")
    run_realtime(CURRENT_ASANA)
