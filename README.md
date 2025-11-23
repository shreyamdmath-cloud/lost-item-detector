Lost Item Detector — AI Vision System 

Project: Lost Item Detector — AI Vision System to Find Lost Objects Using CCTV Video
Team: TRIVYA
Team members: Bhagyashree D Y, Shreya M D, Sneha M U, Vaishnavi A K
Event: Hack-A-Day 2025

This repository contains the working prototype: a Streamlit app that accepts a lost-item image and a CCTV video, scans the video, and returns the top-3 matching frames with bounding boxes and timestamps.

Table of contents

Problem Statement

Current Need & Feasibility

Our Solution

How It Works (Workflow)

What Makes Us Unique

Existing Solutions & Gaps

Tech Stack

System Architecture

What We Built 

Quick Start — Run Locally

Repository Structure

Demo & Expected Inputs

8-Hour Build Plan (summary)

Impact & Future Scope

Final Commit Checklist

Contact

Problem statement

Lost items like bags, phones, and laptops occur daily in public spaces. CCTV footage exists but scanning hours of video manually is slow, error-prone, and inefficient. Security staff and users waste time and resources reviewing footage frame by frame. We build an automated AI tool to rapidly find lost objects in CCTV recordings.

Current need & feasibility

User uploads the lost item image.

User uploads the CCTV video.

Video is automatically converted into frames.

YOLOv8 detects objects in each frame.

ORB extracts feature points from the lost item.

ORB compares those features with detected objects in every frame.

CLIP embedding is used as a backup similarity check.

System calculates a combined matching score for each frame.

Top-3 frames with highest similarity scores are displayed with bounding boxes and timestamps.

This is feasible because modern lightweight models (YOLOv8) and CPU-friendly methods (ORB) allow demo-level performance on normal laptops. A pure-software pipeline means no extra hardware is required.

Our solution 

A working Streamlit web app that:

Accepts an uploaded lost-item image and a CCTV video.

Uses YOLOv8 to detect objects and crop candidate regions.

Extracts ORB key-point descriptors from the lost item (fast, classical features).

Uses CLIP embeddings as a semantic fallback when ORB is unreliable.

Samples frames from the video (configurable sampling rate), runs YOLO on frames, and computes combined similarity scores (ORB + CLIP).

Ranks frames and displays the Top-3 matches with bounding boxes and timestamps.

How it works (workflow)

User uploads lost-item image.

User uploads CCTV video.

Video is converted into frames (sampling every 0.5–1s by default).

YOLOv8 detects objects in each frame and crops candidate regions.

ORB extracts feature points from the lost item crop.

ORB compares key-point descriptors between the lost item and each candidate crop.

CLIP embeddings are computed as backup and used when ORB scores are low.

Combined matching scores are calculated per frame.

System shows Top-3 frames with the highest scores (bounding box + timestamp + score).

What makes us unique

Pure software solution — no special hardware needed.

Works on actual CCTV video (not just single images).

Hybrid matching (YOLO + ORB + CLIP) balances speed and robustness.

Designed for CPU (laptop) demo at hackathon.

Simple, professional Streamlit UI for demo and usability.

Existing solutions & gaps

What exists: manual CCTV playback, enterprise AI surveillance (costly).
Gaps: lack of a simple, affordable, local solution that allows object-based searching through video. Our prototype fills this gap.

Tech stack

Python (core language)

Ultralytics YOLOv8 (yolov8s / yolov8n as appropriate) — object detection

OpenCV — video processing, frame extraction, image ops

ORB — real-time keypoint descriptor & matching (CPU friendly)

CLIP (or a CLIP-like embedding extractor) — semantic image embedding fallback

NumPy — numerical ops & similarity calculations

Streamlit — frontend demo app

System architecture

User Input → YOLO Detection → ORB + CLIP Feature Extraction → Video Frame Extraction → YOLO on Frames → ORB & CLIP Similarity → Ranking → Top-3 Results

(Include diagram in PPT — use the provided Canva/PowerPoint prompt to generate a clean block diagram.)

What we built 
Video upload module

Lost item upload module

YOLO detection & object cropping in the uploaded image

ORB feature extraction for the lost item

CLIP embedding fallback for semantic similarity

Video frame sampling & YOLO detection on frames

ORB matching and CLIP fallback scoring per frame

Ranking to get top 3 frames with bounding boxes and timestamps

Premium Streamlit UI (gradient + banner + Start Search button)
