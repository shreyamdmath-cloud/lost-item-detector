# ===============================================================
#    LOST ITEM DETECTOR - MODERN UI 
# ===============================================================

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

# -------------------------------------------------------------
#                       CUSTOM CSS
# -------------------------------------------------------------
custom_css = """
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #061933, #0a2d57, #0e3b6c);
    color: white;
}

[data-testid="stHeader"] { background: rgba(0,0,0,0); }

.glass-card {
    padding: 22px;
    border-radius: 14px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
}

.upload-label {
    font-size: 20px;
    font-weight: 600;
    color: #E3ECF7;
    margin-bottom: 8px;
}

.banner {
    text-align:center;
    margin-bottom: 10px;
}

</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------------------------------------
#                           BANNER
# -------------------------------------------------------------
st.markdown(
    """
<div class='banner'>
   <img src='https://static.vecteezy.com/system/resources/previews/036/773/480/original/search-icon-logo-design-template-vector.jpg' width='140'>

</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center; color:#F2F6FF;'> Lost Item Detector</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#C9D7E6;'>AI + Video Intelligence to Detect Lost Objects</p>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
#                   STEP 1 — VIDEO UPLOAD
# -------------------------------------------------------------
st.markdown("<div class='upload-label'>Step 1 — Upload CCTV Video</div>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Choose a CCTV video",
        type=["mp4", "mov", "avi", "mkv"],
    )
    video_path = None
    if video_file:
        video_path = os.path.join(UPLOAD_FOLDER, "cctv_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        st.success("Video uploaded successfully!")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
#                STEP 2 — LOST ITEM UPLOAD
# -------------------------------------------------------------
st.markdown("<div class='upload-label'>Step 2 — Upload Lost Item Image</div>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    item_file = st.file_uploader(
        "Upload lost item image",
        type=["jpg", "png", "jpeg"],
    )

    item_path = None
    if item_file:
        item_path = os.path.join(UPLOAD_FOLDER, "lost_item.jpg")
        with open(item_path, "wb") as f:
            f.write(item_file.getbuffer())

        st.image(item_path, caption="Lost Item", width=260)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
#               STEP 3 — RUN SEARCH BUTTON
# -------------------------------------------------------------
start_btn = st.button(
    "🔍 Start Search",
    type="primary",
)

# -------------------------------------------------------------
#          ONLY RUN BACKEND WHEN BUTTON IS CLICKED
# -------------------------------------------------------------
if start_btn:

    if not video_path or not item_path:
        st.error("Please upload both video and lost item image before searching.")
        st.stop()

    st.subheader("Searching for Matches…")
    st.info("Video is being scanned. Please wait...")

    # ----------- Detect lost item ------------------------
    lost_results = yolo.predict(item_path, imgsz=640, conf=0.20, verbose=False)
    boxes = lost_results[0].boxes

    if not boxes:
        st.error("No object detected in the lost item image!")
        st.stop()

    areas = [(int(b.xyxy[0][2] - b.xyxy[0][0]) *
              int(b.xyxy[0][3] - b.xyxy[0][1])) for b in boxes]
    chosen_box = boxes[int(np.argmax(areas))]

    lx1, ly1, lx2, ly2 = map(int, chosen_box.xyxy[0])
    lost_img = cv2.imread(item_path)
    lost_crop = lost_img[ly1:ly2, lx1:lx2]

    lost_gray = cv2.cvtColor(lost_crop, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(lost_gray, None)

    if des1 is None:
        st.error("Feature extraction failed for lost item.")
        st.stop()

    lost_emb = extract_features(lost_crop)

    # ------------------ SCAN VIDEO ------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    frame_interval = int(fps * 1)

    frame_idx = 0
    scores, frames, timestamps, bboxes = [], [], [], []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    progress = st.progress(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        best_bbox = None
        best_score = 0

        if frame_idx % frame_interval == 0:

            det = yolo.predict(frame, imgsz=640, conf=0.20, verbose=False)

            if det and det[0].boxes:
                for b in det[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    obj = frame[y1:y2, x1:x2]

                    gray_obj = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(gray_obj, None)

                    if des2 is not None:
                        matches = bf.match(des1, des2)
                        good = [m for m in matches if m.distance < 40]
                        score_orb = len(good)

                        if score_orb > best_score:
                            best_score = score_orb
                            best_bbox = (x1, y1, x2, y2)

            clip_emb = extract_features(frame)
            if clip_emb is not None:
                clip_score = float(np.dot(lost_emb, clip_emb)) * 100
                best_score = max(best_score, int(clip_score))

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(frame_idx / fps)
            scores.append(best_score)
            bboxes.append(best_bbox)

        frame_idx += 1
        progress.progress(min(100, int(frame_idx / total_frames * 100)))

    cap.release()
    progress.empty()

    # -------------------------------------------------
    #                TOP 3 MATCH RESULTS
    # -------------------------------------------------
    st.subheader("Top Matches")

    top_idxs = np.argsort(scores)[::-1][:3]

    for idx in top_idxs:
        frame = frames[idx].copy()
        bbox = bboxes[idx]

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 227, 255), 3)

        sec = int(timestamps[idx])
        time_str = str(datetime.timedelta(seconds=sec))
        caption = f"Score: {scores[idx]} | Time: {time_str}"

        st.image(frame, caption=caption, width=650)
