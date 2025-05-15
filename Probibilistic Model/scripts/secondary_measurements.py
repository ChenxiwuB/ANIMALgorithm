import argparse, json, os
import cv2
import numpy as np


# relevant landmark indices
JAWLINE_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NOSE_IDXS = [27, 28, 29, 30]
LOWER_NOSE_IDXS = [31, 32, 33, 34, 35]


# --- CLI setup ---
parser = argparse.ArgumentParser(
    description="Compute various eye‐shape metrics for one image."
)
parser.add_argument("--idx",
    help="index of the image in all_data.json (as a string)",
    required=True
)
args = parser.parse_args()

# ---load data---
with open("all_data.json") as f:
    data = json.load(f)

entry = data.get(args.idx)
if entry is None:
    raise SystemExit(f"No entry with index '{args.idx}' in all_data.json")

#----pick out landmarks---
landmark = entry["face_landmarks"]
landmarks = np.array(landmark, dtype=np.float32)

#---------------FACE SHAPE---------------------

def face_shape(landmarks):
    jaw_pts = np.array([landmarks[i] for i in JAWLINE_IDXS],
                    dtype=np.int32).reshape(-1,1,2)

    # fit an ellipse
    (center, axes, angle) = cv2.fitEllipse(jaw_pts)
    # axes is a (width, height) pair; we’ll take:
    major_axis = max(axes)
    minor_axis = min(axes)

    # compute axis‐ratio
    face_shape = minor_axis / major_axis  # ≤ 1

    # classify
    return face_shape


#------------------CHIN LENGTH--------------------
def chin_size(landmarks):
    
    lm = np.asarray(landmarks, dtype=float)

    nose_root = lm[27]
    nose_tip  = lm[30]
    chin_tip  = lm[8]

    nose_length = np.linalg.norm(nose_tip  - nose_root)
    chin_length = np.linalg.norm(chin_tip  - nose_tip)

    chin_size = chin_length / nose_length if nose_length>0 else float('inf')

    print(f" - Chin/Nose ratio: {chin_size:.3f}")
    return chin_size


#------------------EYES----------------------------
def eye_size(landmarks):
    # face width
    face_width = np.linalg.norm(landmarks[16] - landmarks[0])

    # face height
    brow_mid = (landmarks[19] + landmarks[24]) / 2 
    chin_tip = landmarks[8]                       
    face_height = np.linalg.norm(chin_tip - brow_mid)

    #---EYE SIZE(raw)---
    # right eye
    r_width  = np.linalg.norm(landmarks[39] - landmarks[36])
    r_height = (np.linalg.norm(landmarks[37] - landmarks[41]) + np.linalg.norm(landmarks[38] - landmarks[40])) / 2

    # left eye
    l_width  = np.linalg.norm(landmarks[45] - landmarks[42])
    l_height = (np.linalg.norm(landmarks[43] - landmarks[47]) + np.linalg.norm(landmarks[44] - landmarks[46])) / 2

    # normalize to face width&height
    r_w_fw = r_width  / face_width
    l_w_fw = l_width  / face_width

    r_h_fw = r_height / face_width
    l_h_fw = l_height / face_width

    r_w_fh = r_width  / face_height
    l_w_fh = l_width  / face_height

    r_h_fh = r_height / face_height
    l_h_fh = l_height / face_height

    avg_w_fw = (r_w_fw + l_w_fw) / 2
    avg_h_fw = (r_h_fw + l_h_fw) / 2

    avg_w_fh = (r_w_fh + l_w_fh) / 2
    avg_h_fh = (r_h_fh + l_h_fh) / 2

    eye_size = (avg_w_fw + avg_h_fw + avg_w_fh + avg_h_fh) / 4
    return eye_size


#--------------EYE DISTANCE---------------------

face_w = np.linalg.norm(landmarks[16] - landmarks[0])  # face width

def eye_distance(landmarks):
    lm = np.asarray(landmarks, dtype=float)
    fw = np.linalg.norm(lm[16] - lm[0])
    inner_dist = np.linalg.norm(lm[42] - lm[39])
    
    eye_dist = inner_dist / fw
    return eye_dist


#--------------------MOUTH WIDTH------------------------
def mouth_width(landmark):
    left_mouth  = landmarks[48] 
    right_mouth = landmarks[54]
    mouth_width = np.linalg.norm(right_mouth - left_mouth)

    nose_w = np.linalg.norm(landmarks[35] - landmarks[31])
    
    # normalize
    mouth_w = mouth_width / nose_w if nose_w>0 else 0

    print(f" - Mouth width/nose: {mouth_w:.3f}")
    return mouth_w