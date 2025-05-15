import argparse
from primary_measurements import compute_metrics, visualize_face_metrics
from classify import classify
from similarity import disambiguate


parser = argparse.ArgumentParser(
    description="Compute features & classify into animal archetype"
)

parser.add_argument("--idx", required=True)
args = parser.parse_args()


eye, nose, lip, chin = compute_metrics(args.idx)
code, animals = classify(eye, nose, lip, chin)
print(f"Code: {code} | Archetype(s):", ", ".join(animals))

# handle each case
if len(animals) == 1:
    print(f" - {animals[0]}: 100%")
else:
    print("Secondary measurements:")
    # run secondary measurements
    pct_map = disambiguate(args.idx, code)

    # print out percentages
    print("Similarity scores:")
    for animal, pct in sorted(pct_map.items(), key=lambda x: -x[1]):
        print(f" - {animal}: {pct:.1f}%")

    # pick the top‐scoring animal
    best = max(pct_map, key=pct_map.get)
    print("Best guess:", best)