# utils/feature_extract.py
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

# load once (slow on first run while model downloads)
_clip_model = SentenceTransformer("clip-ViT-B-32")  # light + accurate for matching

def _to_pil(img_np):
    """Convert BGR numpy image (OpenCV) to PIL image in RGB."""
    if img_np is None:
        return None
    if isinstance(img_np, np.ndarray):
        # OpenCV uses BGR; convert to RGB
        if img_np.shape[2] == 3:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_np
        return Image.fromarray(img_rgb)
    else:
        # assume path-like
        return Image.open(img_np).convert("RGB")

def extract_features(input_data):
    """
    Input:
      - input_data: either a numpy ndarray (BGR) or a file path string
    Output:
      - normalized 512-d embedding (numpy float32 array)
    """
    pil = _to_pil(input_data)
    if pil is None:
        return None

    # sentence-transformers returns numpy vector
    emb = _clip_model.encode(pil, convert_to_numpy=True, normalize_embeddings=True)
    # ensure float32
    return emb.astype(np.float32)
