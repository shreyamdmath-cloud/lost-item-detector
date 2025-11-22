# app.py

import streamlit as st
import os
import cv2
import numpy as np
import datetime
from ultralytics import YOLO
from utils.feature_extract import extract_features
from utils.similarity import cosine_similarity

# ------------------ LOAD MODELS ------------------
yolo = YOLO("model/yolov8s.pt")
orb = cv2.ORB_create(nfeatures=1500)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🔍 Lost Item Detector – YOLO + CLIP + ORB Hybrid System")

# ------------------ STEP 1: UPLOAD CCTV VIDEO ------------------
st.subheader("Step 1 — Upload CCTV Video")
video_file = st.file_uploader("Upload CCTV video", type=["mp4", "mov", "avi", "mkv"])
video_path = None
if video_file:
    video_path = os.path.join(UPLOAD_FOLDER, "cctv_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.success("Video uploaded.")

# ------------------ STEP 2: UPLOAD LOST ITEM IMAGE ------------------
st.subheader("Step 2 — Upload Lost Item Image")
item_file = st.file_uploader("Upload lost item image", type=["jpg", "png", "jpeg"])
item_path = None
if item_file:
    item_path = os.path.join(UPLOAD_FOLDER, "lost_item.jpg")
    with open(item_path, "wb") as f:
        f.write(item_file.getbuffer())
    st.image(item_path, caption="Lost Item", width=300)
    st.success("Lost item uploaded.")

# ------------------ STEP 3: PROCESS & MATCH ------------------
if video_path and item_path:

    st.subheader("Step 3 — Matching in progress…")
    st.info("Extracting features & scanning frames...")

    # Detect lost item using YOLO
    lost_results = yolo.predict(item_path, imgsz=640, conf=0.20, verbose=False)
    boxes = lost_results[0].boxes
    if boxes is None or len(boxes) == 0:
        st.error("No object detected in the lost item image!")
        st.stop()

    # Choose largest detected object
    areas = [(int(b.xyxy[0][2] - b.xyxy[0][0]) * int(b.xyxy[0][3] - b.xyxy[0][1])) for b in boxes]
    chosen_box = boxes[int(np.argmax(areas))]

    lx1, ly1, lx2, ly2 = map(int, chosen_box.xyxy[0])
    lost_img = cv2.imread(item_path)
    lost_crop = lost_img[ly1:ly2, lx1:lx2]

    # ORB features for lost item
    lost_gray = cv2.cvtColor(lost_crop, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(lost_gray, None)
    if des1 is None:
        st.error("ORB could not extract features from lost item.")
        st.stop()

    # CLIP embedding (backup)
    lost_emb = extract_features(lost_crop)

    # ------------------ SCAN VIDEO ------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    frame_interval = int(fps * 1)

    frame_idx = 0
    scores = []
    frames = []
    timestamps = []
    bboxes = []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pbar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        best_bbox = None
        best_score = 0

        if frame_idx % frame_interval == 0:

            # YOLO detect objects in the frame
            det = yolo.predict(frame, imgsz=640, conf=0.20, verbose=False)

            if det and det[0].boxes:
                for b in det[0].boxes:

                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    obj = frame[y1:y2, x1:x2]

                    # ORB matching
                    gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(gray_obj, None)

                    if des2 is not None:
                        matches = bf.match(des1, des2)
                        good = [m for m in matches if m.distance < 40]
                        score_orb = len(good)

                        if score_orb > best_score:
                            best_score = score_orb
                            best_bbox = (x1, y1, x2, y2)

            # CLIP fallback
            clip_emb = extract_features(frame)
            if clip_emb is not None:
                score_clip = float(np.dot(lost_emb, clip_emb)) * 100
                if score_clip > best_score:
                    best_score = int(score_clip)

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(frame_idx / fps)
            scores.append(best_score)
            bboxes.append(best_bbox)

        frame_idx += 1
        pbar.progress(min(100, int(frame_idx / total_frames * 100)))

    cap.release()
    pbar.empty()

    # ------------------ DISPLAY TOP MATCHES ------------------
    st.subheader("Top 3 Matches")

    top_idxs = np.argsort(scores)[::-1][:3]

    for idx in top_idxs:
        frame = frames[idx].copy()
        bbox = bboxes[idx]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)

        sec = int(timestamps[idx])
        time_str = str(datetime.timedelta(seconds=sec))
        caption = f"Score: {scores[idx]} | Time: {time_str}"

        st.image(frame, caption=caption, width=600)
