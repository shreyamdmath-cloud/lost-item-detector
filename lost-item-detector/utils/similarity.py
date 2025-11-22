# utils/similarity.py
import numpy as np

def cosine_similarity(a, b):
    if a is None or b is None: return 0.0
    # Assuming inputs are normalized embeddings
    return float(np.dot(a, b))
