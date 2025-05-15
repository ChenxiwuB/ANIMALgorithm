import json, os
import cv2
import numpy as np
import math

print("Primary measurements: ")


# landmark indices for each facial feature (68-point dlib)
JAWLINE_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
RIGHT_EYEBROW_IDXS = [17, 18, 19, 20, 21]
LEFT_EYEBROW_IDXS = [22, 23, 24, 25, 26]
NOSE_IDXS = [27, 28, 29, 30]
LOWER_NOSE_IDXS = [31, 32, 33, 34, 35]
RIGHT_EYE_IDXS = [36, 37, 38, 39, 40, 41]
LEFT_EYE_IDXS = [42, 43, 44, 45, 46, 47]
OUTER_LIP_IDXS = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
INNER_LIP_IDXS = [60, 61, 62, 63, 64, 65, 66, 67]


def load_landmarks(idx, data_path="all_data.json"):
    """
    Return the raw 68 (x,y) landmarks for image `idx`.
    """
    with open(data_path) as f:
        data = json.load(f)
    entry = data.get(idx)
    if entry is None:
        raise ValueError(f"No entry for index '{idx}'")
    return entry["face_landmarks"]


#----------------------EYES--------------------------------------------------
def ellipse_roundness(pts):
    arr = np.array(pts, dtype=np.int32).reshape(-1,1,2)
    (_, _), (major, minor), _ = cv2.fitEllipse(arr)
    return min(major, minor) / max(major, minor)

def eye_circularity(pts):
    p = np.array(pts, float)
    x, y = p[:,0], p[:,1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    perim = np.sum(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1))
    return (4 * math.pi * area / (perim*perim)) if perim > 0 else 0


#-----------------NOSE------------------------------------------------
def nose_width_length(landmarks, nose_idxs):
    # center points
    pts = np.array([landmarks[i] for i in nose_idxs], dtype=float)
    mean = pts.mean(axis=0)
    centered = pts - mean

    # nose axis vector
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]
    axis = axis / np.linalg.norm(axis)

    # perpendicular unit vector
    perp = np.array([-axis[1], axis[0]])

    # length
    proj = centered.dot(axis)
    length = proj.max() - proj.min()

    # width
    perp_vals = centered.dot(perp)
    width = perp_vals.max() - perp_vals.min()

    return {
        "length_width_ratio": length/width if width>0 else 0.0
    }


#------------------LIPS----------------------
# lip fullness
def lip_width(landmarks):
    #length
    xL, yL = landmarks[48]
    xR, yR = landmarks[54]
    width_px = math.hypot(xR - xL, yR - yL)

    # height
    top_outer = np.mean([landmarks[50], landmarks[52]], axis=0)
    bot_outer = np.array(landmarks[57], dtype=float)
    height_px = math.hypot(*(bot_outer - top_outer))

    raw = height_px/width_px
    
    return {
        "raw": raw
    }


#------------------CHIN--------------------
def point_curvature(p_prev, p_curr, p_next):
    a = np.linalg.norm(p_next - p_curr)
    b = np.linalg.norm(p_curr - p_prev)
    c = np.linalg.norm(p_next - p_prev)
    s = (a + b + c) / 2
    area = math.sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
    if a*b*c == 0:
        return 0.0
    return 4 * area / (a * b * c)

def chin_curvature_score(landmarks, max_k):
    
    jaw = np.array([landmarks[i] for i in JAWLINE_IDXS], float)
    # relevant points
    chin_positions = [6, 7, 8, 9, 10]
    
    # compute curvature for each
    local_ks = []
    for i in chin_positions:
        p_prev = jaw[i-1]
        p_curr = jaw[i]
        p_next = jaw[i+1]
        local_ks.append(point_curvature(p_prev, p_curr, p_next))
    
    mean_k = float(np.mean(local_ks))

    roundness = (max_k - mean_k) / max_k
    roundness = max(0.0, min(1.0, roundness))

    return {
        "chin_roundness": roundness
    }


#---------------------------compute all----------------------------------
def compute_metrics(idx, data_path="all_data.json", img_dir="images"):
    # load JSON and pick the right entry
    with open(data_path) as f:
        data = json.load(f)
    entry = data.get(idx)
    if entry is None:
        raise ValueError(f"No entry for index '{idx}'")
    landmarks = entry["face_landmarks"]

    # load image
    img_path = os.path.join(img_dir, entry["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Can't load image at {img_path}")

    # compute eye metric
    r_pts = [landmarks[i] for i in RIGHT_EYE_IDXS]
    l_pts = [landmarks[i] for i in LEFT_EYE_IDXS]
    r_round = ellipse_roundness(r_pts)
    l_round = ellipse_roundness(l_pts)
    r_circ  = eye_circularity(r_pts)
    l_circ  = eye_circularity(l_pts)
    avg_hybrid2 = ((r_round + r_circ)/2 + (l_round + l_circ)/2) / 2
    print(f" - EYES: Average 2-metric hybrid: {avg_hybrid2:.3f}")

    # compute nose metric
    fnose_ratio = nose_width_length(landmarks, NOSE_IDXS + LOWER_NOSE_IDXS)[
        "length_width_ratio"
    ]
    print(f" - NOSE: length/width ratio = {fnose_ratio:.3f}")

    # compute lip metric
    flip_ratio  = lip_width(landmarks)["raw"]
    print(f" - LIPS: height/width = {flip_ratio:.3f}")

    # compute chin metric
    fchin_round = chin_curvature_score(landmarks, max_k=0.04) [
        "chin_roundness"
    ]
    print(f" - CHIN: roundness = {fchin_round:.3f}")
    
    visualize_face_metrics(landmarks, img)

    return avg_hybrid2, fnose_ratio, flip_ratio, fchin_round


#------------ used in secondary_measurements--------------------------

def nose_ratio(lm):
    return nose_width_length(lm, NOSE_IDXS+LOWER_NOSE_IDXS)["length_width_ratio"]

def lip_ratio(lm):
    return lip_width(lm)["raw"]

def chin_roundness(lm):
    return chin_curvature_score(lm, max_k=0.04)["chin_roundness"]


#----------------------------visualize------------------------------------
def visualize_face_metrics(landmarks, image):
    img = image.copy() if image is not None else np.zeros((max(int(y) for _,y in landmarks)+20,
                                                          max(int(x) for x,_ in landmarks)+20,3), np.uint8)
    # Eyes
    for eye_idxs in (RIGHT_EYE_IDXS, LEFT_EYE_IDXS):
        pts = np.array([landmarks[i] for i in eye_idxs], np.int32).reshape(-1,1,2)
        cv2.polylines(img, [pts], True, (0,255,0), 1)
        (cx,cy),(maj,min_),ang = cv2.fitEllipse(pts)
        cv2.ellipse(img,(int(cx),int(cy)),(int(maj/2),int(min_/2)),ang,0,360,(0,255,255),2)
        theta = math.radians(ang)
        dx,dy = (maj/2)*math.cos(theta),(maj/2)*math.sin(theta)
        cv2.line(img,(int(cx-dx),int(cy-dy)),(int(cx+dx),int(cy+dy)),(0,0,255),2)
        dx2,dy2 = (min_/2)*math.cos(theta+math.pi/2),(min_/2)*math.sin(theta+math.pi/2)
        cv2.line(img,(int(cx-dx2),int(cy-dy2)),(int(cx+dx2),int(cy+dy2)),(255,0,0),2)
        hull = cv2.convexHull(pts)
        cv2.polylines(img,[hull],True,(0,255,255),1)
    
    # Nose 
    pts_n = np.array([landmarks[i] for i in NOSE_IDXS+LOWER_NOSE_IDXS],float)
    m = pts_n.mean(axis=0); ctr = pts_n-m
    _,_,Vt = np.linalg.svd(ctr,full_matrices=False); axis=Vt[0]/np.linalg.norm(Vt[0]); perp=np.array([-axis[1],axis[0]])
    proj = ctr.dot(axis); start_n,end_n = m+axis*proj.min(), m+axis*proj.max()
    pv = ctr.dot(perp); start_w,end_w = m+perp*pv.min(), m+perp*pv.max()
    cv2.line(img,tuple(start_n.astype(int)),tuple(end_n.astype(int)),(0,0,255),2)
    cv2.line(img,tuple(start_w.astype(int)),tuple(end_w.astype(int)),(255,0,0),2)
    
    # Lips
    L = np.array(landmarks[48],float); R = np.array(landmarks[54],float)
    top = np.array(landmarks[51], float); bot = np.array(landmarks[57],float)
    cv2.line(img,tuple(L.astype(int)),tuple(R.astype(int)),(0,255,255),2)
    cv2.line(img,tuple(top.astype(int)),tuple(bot.astype(int)),(255,0,255),2)
    
    # Chin
    jaw_pts = np.array([landmarks[i] for i in JAWLINE_IDXS],np.int32).reshape(-1,1,2)
    cv2.polylines(img,[jaw_pts],False,(0,165,255),2)
    
    cv2.imshow("Face Shape + Metrics",img); cv2.waitKey(0); cv2.destroyAllWindows()



    