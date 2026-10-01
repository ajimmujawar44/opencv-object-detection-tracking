# 🚗 Highway Vehicle Detection & Object Tracking

A computer vision project built with Python and OpenCV to detect moving objects, track them across video frames, and perform white-color masking on highway video footage.

The project processes a highway video and provides multiple computer vision operations through a simple command-line menu.

---

## 📌 Project Overview

This project demonstrates fundamental computer vision techniques for video analysis.

The system can:

- 🚗 Detect moving objects from video
- 🎯 Track detected objects using unique IDs
- ⚪ Detect white regions using HSV color masking
- 🎥 Save processed videos as output
- 📱 Convert landscape video into a portrait-style 9:16 view

The project is designed with a modular structure so that detection, tracking, and masking logic are separated into different files.

---

## ✨ Features

### 1. Object Detection

Uses OpenCV background subtraction and contour detection to identify moving objects in the highway video.

Main techniques:

- MOG2 Background Subtraction
- Thresholding
- Morphological Operations
- Contour Detection
- Bounding Boxes

---

### 2. Object Tracking

Detected objects are tracked between video frames using a custom centroid-based tracker.

Each tracked object receives a unique ID:

```text
ID 0
ID 1
ID 2
...