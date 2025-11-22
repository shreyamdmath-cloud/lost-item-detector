
import numpy as np

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0
    return dot / norm
