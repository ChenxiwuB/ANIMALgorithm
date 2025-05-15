import numpy as np
from secondary_measurements import face_shape, mouth_width, eye_size, chin_size, eye_distance
from primary_measurements import load_landmarks, nose_ratio, lip_ratio, chin_roundness

animal_pools = {
    "AAAA": ["Dog", "Deer", "Hamster", "Turtle"],
    "AABA": ["Hamster", "Turtle"],
    "AABB": ["Dog", "Deer"],
    "ABAA": ["Dog", "Deer", "Bunny"],
    "ABAB": ["Fox", "Snake", "Bunny"],
    "ABBA": ["Fish"],
    "ABBB": ["Dog", "Cat", "Deer"],
    "BAAA": ["Cat", "Fish"],
    "BABA": ["Cat"],
    "BABB": ["Cat"],
    "BAAB": ["Bird"],
    "AAAB": ["Fox", "Snake", "Bird"],
    "BBAB": ["Fox", "Snake", "Bird"],
    "BBBB": ["Fox", "Snake"],
    "BBAA": ["Dog", "Deer"],
    "BBBA": ["Fish"],
}


SECONDARY_FEATURES = {
  "AAAA": [face_shape, mouth_width, eye_size],
  "AABA": [face_shape, mouth_width, eye_size],
  "AABB": [face_shape],
  "ABAA": [face_shape, mouth_width],
  "ABAB": [mouth_width, nose_ratio, face_shape, chin_roundness, lip_ratio],
  "ABBB": [face_shape, chin_size],
  "BAAA": [chin_size, eye_distance],
  "AAAB": [mouth_width, nose_ratio, face_shape, chin_roundness, lip_ratio],
  "BBAB": [mouth_width, nose_ratio, face_shape, chin_roundness, lip_ratio],
  "BBBB": [nose_ratio, face_shape, chin_roundness, lip_ratio],
  "BBAA": [face_shape]
}

PROTOTYPES = {
  "Dog": {"face_shape":0.76, "mouth_width": 0.200, "eye_size": 0.115, "lip_ratio": 0.38, "chin_size": 2.1},
  "Deer": {"face_shape":0.7, "mouth_width": 0.200, "eye_size": 0.115, "lip_ratio": 0.38, "chin_size": 2.1},
  "Hamster": {"face_shape":0.78, "mouth_width": 0.195, "eye_size": 0.115},
  "Turtle": {"face_shape":0.77, "mouth_width": 0.235, "eye_size": 0.125},
  "Bunny": {"face_shape":0.76, "mouth_width": 0.165, "nose_ratio": 1.750, "chin_roundness": 0.780, "lip_ratio": 0.4},
  "Fox": {"face_shape": 0.65, "mouth_width": 0.195, "nose_ratio": 1.8, "chin_roundness": 0.7, "lip_ratio": 0.38},
  "Snake": {"face_shape": 0.6, "mouth_width": 0.200, "nose_ratio": 1.8, "chin_roundness": 0.77, "lip_ratio": 0.32},
  "Cat": {"face_shape":0.76, "chin_size": 1.4, "eye_distance": 0.253},
  "Fish": {"chin_size": 2.1, "eye_distance": 0.273},
  "Bird": {"mouth_width": 0.142, "nose_ratio": 1.78, "lip_ratio": 0.39, "chin_roundness": 0.77, "face_shape": 0.65}
}

def disambiguate(idx, code):
    fns = SECONDARY_FEATURES[code]
    candidates = animal_pools[code]

    lm = load_landmarks(idx)
    feats = { fn.__name__: fn(lm) for fn in fns }

    protos = {
      a: { feat: PROTOTYPES[a][feat] for feat in feats }
      for a in candidates
    }

    return similarity_percentages(protos, feats)

def similarity_percentages(prototypes, features, eps=1e-6):

    animals = list(prototypes.keys())
    feat_names = list(features.keys())
    # build arrays
    P = np.array([[prototypes[a][f] for f in feat_names] for a in animals])
    F = np.array([features[f] for f in feat_names])

    # euclidean distances
    d = np.linalg.norm(P - F[np.newaxis,:], axis=1)

    # invert distances to similarities
    s = 1.0 / (d + eps)

    # normalize to percentages
    total = s.sum()
    pct = { animals[i]: float(s[i] / total * 100) for i in range(len(animals)) }
    return pct