# Pothole Detection & Severity Assessment System 🛣️

An end-to-end computer vision and web deployment pipeline designed to identify road hazards, estimate physical depth from 2D images, and calculate structural volume for automated severity classification. 

Developed as an academic project at **Galgotias University** by **Uttam Verma**.

---

## 🧠 Project Overview
Standard pothole detection systems typically stop at 2D bounding boxes or simple masks. This project advances the methodology by integrating **Sensor Fusion** in a purely monocular vision environment. By combining custom-trained semantic segmentation with zero-shot monocular depth estimation, this system extracts approximate Z-axis measurements (depth and volume) from standard dashcam or smartphone imagery without requiring hardware LiDAR.

## ✨ Core Features & Architecture

*   **Custom Semantic Segmentation (X/Y Axis):** Utilizes a fine-tuned NVIDIA `mit-b0` SegFormer model (trained over 50 epochs). The model features specialized `ReflectionAugmentation` during training, allowing it to accurately map the jagged edges of water-filled potholes while ignoring specular glare and sky reflections.
*   **Monocular Depth Estimation (Z-Axis):** Integrates Intel's `dpt-swinv2-tiny-256` (MiDaS) to generate dense disparity maps, intelligently evaluating shadows and perspective to estimate the relative depth of the hazard.
*   **Optical Anomaly Override (Guardrail System):** Implements dynamic heuristic clamping. If the depth estimator is blinded by a mirrored sky reflection in a puddle (resulting in an "infinite depth" anomaly >25cm), the Python backend intercepts the error and normalizes the data to a realistic critical threshold (7.5cm - 12.0cm).
*   **Civil Engineering Logic Engine:** Converts pixel surface area and depth disparity into physical cubic meters (m³). Classifies the hazard into actionable tiers (MINOR, MODERATE, CRITICAL) based on volume thresholds (e.g., > 0.05 m³ = Immediate Repair).
*   **Interactive Web UI:** A locally hosted Flask & Bootstrap dashboard that processes user uploads and renders segmented overlays alongside exact severity metrics.

## 🛠️ Technology Stack
*   **Deep Learning:** PyTorch, Hugging Face