# SafeCityAI 🚦

## AI-Based Traffic Safety Violation Detection

SafeCityAI is an object detection system designed to automatically identify traffic-safety-related objects and violations from images and video.

The system uses a custom-trained YOLO object detection model and provides both video inference and a FastAPI REST API for image-based detection.

---

## 🎯 Problem Statement

Traffic monitoring through manual surveillance is time-consuming and difficult to scale.

SafeCityAI uses computer vision to automatically analyze traffic imagery and identify safety-related objects such as helmets, no-helmet cases, and license plates.

---

## 🚀 Features

- Custom-trained YOLO object detection model
- Helmet detection
- No-Helmet detection
- License Plate detection
- Image-based inference
- Video-based inference
- Annotated output video generation
- FastAPI REST API
- JSON-based detection results
- Confidence scores
- Bounding-box coordinates

---

## 🧠 System Architecture

```text
Traffic Image / Video
        ↓
Frame Extraction
        ↓
Custom YOLO Model
        ↓
Object Detection
        ↓
Bounding Boxes + Confidence
        ↓
┌─────────────────────┐
│                     │
▼                     ▼
Annotated Video    FastAPI REST API
                      ↓








