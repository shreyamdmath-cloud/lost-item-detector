import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.feature_extract import extract_features
from utils.similarity import cosine_similarity

model = YOLO("model/yolov8n.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title("🔍 Lost Item Detector – AI Vision System (Video + Image Matching)")
st.write("Upload a CCTV video and a lost item image. The system will scan the video and show top 3 matching frames.")

# ---------------------------------------------------------
# STEP 1: UPLOAD CCTV VIDEO
# ---------------------------------------------------------
st.subheader("Step 1: Upload CCTV Video")
video_file = st.file_uploader("Upload a CCTV video", type=["mp4", "mov", "avi", "mkv"])

# Store video temporarily
video_path = None
if video_file:
    video_path = os.path.join(UPLOAD_FOLDER, "cctv_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.success("Video uploaded successfully!")

# ---------------------------------------------------------
# STEP 2: UPLOAD LOST ITEM IMAGE
# ---------------------------------------------------------
st.subheader("Step 2: Upload Lost Item Image")
item_file = st.file_uploader("Upload lost item photo", type=["jpg", "png", "jpeg"])

item_path = None
if item_file:
    item_path = os.path.join(UPLOAD_FOLDER, "lost_item.jpg")
    with open(item_path, "wb") as f:
        f.write(item_file.getbuffer())
    st.image(item_path, caption="Lost Item", width=300)
    st.success("Lost item image uploaded!")

# ---------------------------------------------------------
# STEP 3: PROCESS VIDEO & MATCH
# ---------------------------------------------------------
if video_path and item_path:
    st.subheader("Step 3: Finding Matches…")
    st.info("Extracting frames and comparing… please wait.")

    # Extract feature for lost item
    query_features = extract_features(item_path)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * 3  # every 3 seconds
    frame_count = 0

    scores = []
    frame_numbers = []
    frames_data = []

    index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # TEMP: save frame to a numpy array (not disk)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract features from this frame
            frame_features = extract_features(frame)

            # Compute similarity
            score = cosine_similarity(query_features, frame_features)

            # Save match info
            scores.append(score)
            frame_numbers.append(index)
            frames_data.append(rgb)

            index += 1

        frame_count += 1

    cap.release()

    # Sort and find top 3 matches
    top_indices = np.argsort(scores)[::-1][:3]

    st.subheader("🎯 Top 3 Matching Frames")

    if max(scores) < 0.1:
        st.error("No matching object found in the entire video.")
    else:
        for rank, idx in enumerate(top_indices):
            st.image(frames_data[idx], caption=f"Match Score: {round(scores[idx], 3)}", width=400)
