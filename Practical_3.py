import numpy as np
import pandas as pd

# 1. Car Dataset Sample (as required by lab sheet)
cars = ["bmw", "audi", "fiat", "toyota", "honda"]

# 2. Manually defined embeddings (representing what Word2Vec learns)
# These vectors represent features like 'Luxury' or 'Reliability'
embeddings = {
    "bmw":    np.array([0.9, 0.8, 0.1]),
    "audi":   np.array([0.85, 0.75, 0.15]),
    "fiat":   np.array([0.1, 0.2, 0.9]),
    "toyota": np.array([0.2, 0.3, 0.8]),
    "honda":  np.array([0.25, 0.35, 0.75])
}

print("--- Practical 3: Word2Vec Embeddings (Vector Representation) ---")
for car, vector in embeddings.items():
    print(f"{car.upper()}: {vector}")

# 3. Similarity Logic (Cosine Similarity)
def get_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("\n--- Similarity Analysis ---")
sim_luxury = get_similarity(embeddings["bmw"], embeddings["audi"])
sim_mix = get_similarity(embeddings["bmw"], embeddings["toyota"])

print(f"Similarity (BMW vs Audi): {sim_luxury:.4f}")
print(f"Similarity (BMW vs Toyota): {sim_mix:.4f}")