Lost Item Detector — AI Vision System
Hack-A-Day 2025 | Team TRIVYA

Team Members:

Bhagyashree D Y

Shreya M D

Sneha M U

Vaishnavi A K

A complete AI-powered system to detect lost objects in CCTV videos using YOLOv8, ORB feature matching, and CLIP embeddings.
This repository contains the fully working Streamlit application.

 Table of Contents

Problem Statement

Current Need & Feasibility

Our Solution

How It Works — Workflow

What Makes Us Unique

Existing Solutions & Gaps

Tech Stack

System Architecture

What We Built

Quick Start — Run Locally

Repository Structure

Demo & Expected Inputs

8-Hour Build Plan (Summary)

Impact & Future Scope

Final Commit Checklist

Contact

 Problem Statement

People frequently lose items like bags, wallets, phones, bottles etc.
CCTV recordings contain evidence, but manually checking hours of footage is slow and inefficient.

Security teams waste:

Time

Manpower

Resources

Goal: Build an AI system to automatically find the lost item in CCTV video.

Current Need & Feasibility

This solution is practical because:

YOLOv8 is lightweight & accurate

ORB is fast and CPU-friendly

CLIP embeddings improve robustness

Entire pipeline runs on any laptop

 Our Solution

A working Streamlit app that:

Accepts lost item image

Accepts CCTV video

Uses YOLOv8 to detect objects

Extracts ORB keypoints

Uses CLIP embedding for semantic fallback

Scores similarity

Shows Top 3 matching frames with bounding boxes & timestamps

 How It Works — Workflow

User uploads lost-item image

User uploads CCTV video

Video → converted into frames

YOLO detects objects in each frame

ORB extracts features from the lost item

ORB compares features with objects in frames

CLIP embedding acts as backup

Similarity score calculated

Top-3 frames are shown

 What Makes Us Unique

Hybrid model (YOLO + ORB + CLIP)

Pure software solution

Works on normal laptop

Real CCTV video supported

Professional Streamlit UI

Lightweight & hackathon-friendly

 Existing Solutions & Gaps

Existing: Manual CCTV review, expensive enterprise surveillance tools
Gaps:

No simple & affordable local system

No object-based search

No open-source version

Our prototype fills the gap.

 Tech Stack

Python

Ultralytics YOLOv8

OpenCV

ORB keypoint extraction

CLIP embedding

NumPy

Streamlit
 System Architecture
User Input (Image + Video)
        ↓
YOLO Detection (Lost Item)
        ↓
ORB + CLIP Feature Extraction
        ↓
Video Frame Extraction
        ↓
YOLO on Frames
        ↓
ORB & CLIP Similarity
        ↓
Ranking Algorithm
        ↓
Top-3 Results (Frames + Boxes + Timestamps)

What We Built

Video upload module

Lost item upload module

YOLO detection on uploaded image

ORB feature extraction

CLIP embedding fallback

Frame sampler

Matching engine

Ranking logic

Top-3 results display

Premium dark UI

⚙️ Quick Start — Run Locally
pip install -r requirements.txt
streamlit run app.py

 Repository Structure
 lost-item-detector
│── app.py
│── requirements.txt
│── model/
│   └── yolov8s.pt
│── utils/
│   ├── feature_extract.py
│   └── similarity.py
│── uploads/
│── README.md

Demo & Expected Inputs

Lost item:

Clear, close-up photo

JPG/PNG

CCTV video:

MP4/MOV/AVI

10–60 seconds recommended

Good lighting improves results

 8-Hour Build Plan (Summary)
Hour	Task
1	Problem understanding + architecture
2	YOLO integration
3	ORB matching
4	Frame extraction
5	CLIP fallback
6	Scoring system
7	Streamlit UI
8	Testing + PPT + GitHub
 Impact & Future Scope

Multi-camera lost-item tracking

Real-time CCTV support

Improved accuracy via fine-tuned models

Mobile app version

Alert/notification system

 Final Commit Checklist

 Code pushed

 README added

 Streamlit UI working

 Video upload tested

 YOLO model included

 Documentation prepared

 Final PPT ready

 Contact

Team TRIVYA – Hack-A-Day 2025
For queries, connect during event hours.
