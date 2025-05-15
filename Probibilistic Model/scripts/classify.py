import argparse
from primary_measurements import compute_metrics

animal_pools = {
    "Dog":     ["AAAA","AABB","ABAA","ABBB","BBAA"],
    "Cat":     ["ABBB","BAAA","BABA","BABB"],
    "Fox":     ["AAAB","ABAB","BBAB","BBBB"],
    "Snake":   ["AAAB","ABAB","BBAB","BBBB"],
    "Bird":    ["AAAB","BAAB","BBAB"],
    "Deer":    ["AAAA","AABB","ABAA","ABBB","BBAA"],
    "Bunny":   ["ABAA","ABAB"],
    "Hamster": ["AAAA","AABA"],
    "Turtle":  ["AAAA","AABA"],
    "Fish":    ["ABBA","BAAA","BBBA"],
}

# thresholds for each feature
EYE_THRESH  = 0.495  #  narrow(B) < _EYE_ < round(A)
NOSE_THRESH = 1.760  # long(B) < NOSE < wide(A)
LIP_THRESH  = 0.395  # thin(A) < LIP < full(B)
CHIN_THRESH = 0.770  # sharp(B) < CHIN < round(A)

def code_from_metrics(eye, nose, lip, chin):
    """ 4‐letter code."""
    e = "A" if eye > EYE_THRESH  else "B"
    n = "A" if nose < NOSE_THRESH else "B"
    l = "A" if lip < LIP_THRESH  else "B"
    c = "A" if chin > CHIN_THRESH else "B"
    return e + n + l + c

def classify(eye, nose, lip, chin):
    code = code_from_metrics(eye, nose, lip, chin)
    matches = [animal for animal, codes in animal_pools.items() if code in codes]
    return code, matches

if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Classify one face into archetype")
    p.add_argument("--idx", required=True, help="index in all_data.json")
    args = p.parse_args()

    eye, nose, lip, chin = compute_metrics(args.idx)
    code, animals   = classify(eye, nose, lip, chin)

    print(f"4-letter code: {code}")
    print("Matching archetype(s):", ", ".join(animals))
   