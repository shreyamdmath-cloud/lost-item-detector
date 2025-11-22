# app.py
import streamlit as st
import os
import cv2
import numpy as np
import datetime
from ultralytics import YOLO
from utils.feature_extract import extract_features
from utils.similarity import cosine_similarity  # normalized embeddings = dot product

# ------------------ LOAD YOLO MODEL (strong model) ------------------
yolo = YOLO("model/yolov8s.pt")     # better detection than nano

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🔍 Lost Item Detector – CLIP Embeddings + YOLO + Fallback Scan")
st.write("Upload CCTV video and lost item image. System detects object class and performs CLIP-based matching with fallback scanning.")

# ------------------ STEP 1: UPLOAD CCTV VIDEO ------------------
st.subheader("Step 1 — Upload CCTV Video")
video_file = st.file_uploader("Upload CCTV video (MP4/MOV/AVI/MKV)", type=["mp4","mov","avi","mkv"])

video_path = None
if video_file:
    video_path = os.path.join(UPLOAD_FOLDER, "cctv_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.success("Video uploaded.")

# ------------------ STEP 2: UPLOAD LOST ITEM ------------------
st.subheader("Step 2 — Upload Lost Item Image")
item_file = st.file_uploader("Upload lost item image (jpg/png/jpeg)", type=["jpg","png","jpeg"])

item_path = None
if item_file:
    item_path = os.path.join(UPLOAD_FOLDER, "lost_item.jpg")
    with open(item_path, "wb") as f:
        f.write(item_file.getbuffer())
    st.image(item_path, caption="Lost item (input)", width=300)
    st.success("Lost item uploaded.")

# ------------------ STEP 3: PROCESS & MATCH ------------------
if video_path and item_path:
    st.subheader("Step 3 — Searching for matches (please wait)")
    st.info("Detecting objects and scanning frames...")

    # Detect object in lost item image
    item_results = yolo.predict(item_path, imgsz=640, conf=0.05, iou=0.1, verbose=False)
    boxes = item_results[0].boxes

    if boxes is None or len(boxes) == 0:
        st.error("No object detected in the lost item image! Provide a clearer crop.")
        st.stop()

    # Pick largest detected box
    areas = [(int(b.xyxy[0][2] - b.xyxy[0][0]) * int(b.xyxy[0][3] - b.xyxy[0][1])) for b in boxes]
    largest_idx = int(np.argmax(areas))
    chosen_box = boxes[largest_idx]

    lost_class_id = int(chosen_box.cls[0])
    lost_class_name = yolo.names[lost_class_id]

    x1, y1, x2, y2 = map(int, chosen_box.xyxy[0])
    img_bgr = cv2.imread(item_path)
    lost_crop = img_bgr[y1:y2, x1:x2]

    st.write(f"Detected lost item class: **{lost_class_name}**")

    # CLIP embedding for lost item
    lost_emb = extract_features(lost_crop)

    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    SAMPLE_SEC = 0.5                     # more frames = higher accuracy
    IMGSZ = 1280                          # high resolution YOLO input
    CONF = 0.05                           # low conf for small objects
    IOU = 0.1                             # loose NMS for small boxes

    frame_interval = int(max(1, fps * SAMPLE_SEC))
    frame_idx = 0

    scores = []
    frames_rgb = []
    timestamps = []

    pbar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    est_samples = total_frames // frame_interval if total_frames > 0 else None
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            best_score_for_frame = 0.0
            best_bbox = None

            # ------------------ YOLO OBJECT MATCH ------------------
            det = yolo.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)

            if det and det[0].boxes is not None:
                for b in det[0].boxes:
                    cls_id = int(b.cls[0])
                    if cls_id != lost_class_id:
                        continue

                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    if (bx2 - bx1) < 15 or (by2 - by1) < 15:
                        continue

                    crop = frame[by1:by2, bx1:bx2]
                    crop_emb = extract_features(crop)
                    score = float(np.dot(lost_emb, crop_emb))

                    if score > best_score_for_frame:
                        best_score_for_frame = score
                        best_bbox = (bx1, by1, bx2, by2)

            # ------------------ FALLBACK TILE SCAN ------------------
            if best_score_for_frame < 0.05:
                H, W = frame.shape[:2]
                GRID = 4

                tile_h = H // GRID
                tile_w = W // GRID

                for gy in range(GRID):
                    for gx in range(GRID):
                        tx1 = gx * tile_w
                        ty1 = gy * tile_h
                        tx2 = min(W, tx1 + tile_w)
                        ty2 = min(H, ty1 + tile_h)

                        if (tx2 - tx1) < 30 or (ty2 - ty1) < 30:
                            continue

                        tile = frame[ty1:ty2, tx1:tx2]
                        tile_emb = extract_features(tile)
                        score = float(np.dot(lost_emb, tile_emb))

                        if score > best_score_for_frame:
                            best_score_for_frame = score
                            best_bbox = (tx1, ty1, tx2, ty2)

            # Draw bounding box if found
            if best_bbox:
                x1b, y1b, x2b, y2b = best_bbox
                cv2.rectangle(rgb, (x1b, y1b), (x2b, y2b), (255, 0, 0), 3)

            # Store results
            scores.append(best_score_for_frame)
            frames_rgb.append(rgb)
            timestamps.append(frame_idx / fps)

            processed += 1
            if est_samples:
                pbar.progress(min(100, int(processed / est_samples * 100)))

        frame_idx += 1

    cap.release()
    pbar.empty()

    # ------------------ DISPLAY TOP MATCHES ------------------
    if len(scores) == 0:
        st.error("No frames processed.")
        st.stop()

    top_n = min(3, len(scores))
    top_indices = np.argsort(scores)[::-1][:top_n]

    THRESH = 0.20        # lower threshold for metal bottle detection
    st.subheader("Top Matches")

    any_above = False
    for idx in top_indices:
        score = float(scores[idx])
        timestamp = str(datetime.timedelta(seconds=int(timestamps[idx])))
        caption = f"Score: {score:.3f} | Time: {timestamp}"

        if score >= THRESH:
            any_above = True
        else:
            caption += " (low match)"

        st.image(frames_rgb[idx], caption=caption, width=640)

    if not any_above:
        st.warning("No strong matches found. The object may be very small or unclear in the video.")
