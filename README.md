<img width="615" height="531" alt="image" src="https://github.com/user-attachments/assets/490db8c7-9ffe-4284-862b-c30e01b0a57a" /># AI Colony Detection System (Colony-Count-YOLO)

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLO](https://img.shields.io/badge/AI-YOLOv8%2Fv11-purple)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Vue.js](https://img.shields.io/badge/Frontend-Vue.js%203-4FC08D)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)


---

## Executive Summary
Counting bacterial colonies on petri dishes is a fundamental but tedious task in microbiology. Manual counting is time-consuming and prone to human error.

**Colony-Count-YOLO** is an end-to-end web application that automates this process using computer vision. It allows users to upload images, receive instant counts with visual bounding boxes, manually correct the results (Human-in-the-loop), and export reports. The system is containerized for easy deployment and supports GPU acceleration.

---

## Visual Demonstration

### 1. Detection Results (Before vs. After)
The model effectively detects colonies even in challenging lighting conditions or with high density.

![Comparison Result](<img width="615" height="531" alt="image" src="https://github.com/user-attachments/assets/6e435056-4e6a-47b4-a5d3-bdd80f631ded" />
)
*Left: Raw Input Image | Right: AI Detection Output (Count: 142)*

### 2. User Interface Workflow
The web interface is designed for intuitive usage by lab technicians.

| Image Upload | Interactive Detection & Editing |
| :---: | :---: |
| ![Upload UI](assets/images/ui_upload.png) | ![Result UI](assets/images/ui_result.png) |
| *Simple drag-and-drop interface* | *Canvas-based editor for manual corrections* |

---

## System Architecture
The system follows a microservices-like architecture, separating the heavy AI inference from the client-side interaction.

![Architecture Diagram](assets/images/system_architecture.png)

**Core Components:**
* **Frontend (Vue.js / Nuxt):** Handles image uploads and utilizes HTML5 Canvas for an interactive bounding-box editor (allowing users to add/delete/resize boxes).
* **Backend (FastAPI):** Asynchronous API that manages the inference pipeline, image storage, and database transactions.
* **AI Engine (Ultralytics YOLO):** Runs inference on the GPU, featuring custom logic for Non-Maximum Suppression (NMS) tuning.
* **Database (SQLite/PostgreSQL):** Stores prediction history, metadata, and user-edited counts for future model fine-tuning.

---

## AI Methodology & Training

### 1. Dataset Preparation
* **Data Collection:** Custom dataset of petri dish images.
* **Annotation:** Labeled using standard YOLO format (normalized coordinates).
* **Augmentation Strategy:** To ensure robustness against lighting variations, the following augmentations were applied during training:
    * *CLAHE (Contrast Limited Adaptive Histogram Equalization)*
    * *Random Brightness/Contrast*
    * *Mosaic & Mixup* (to handle overlapping colonies)
    * *Random Flip/Rotation*

### 2. Model Training
Trained using **YOLOv8 / YOLOv11** architectures via Transfer Learning.

* **Hyperparameters:**
    * `imgsz`: 832 (High resolution for small objects)
    * `epochs`: 100+ (with Early Stopping)
    * `batch`: 16
    * `optimizer`: AdamW

---

## Performance & Evaluation

The model demonstrates high precision in identifying individual colonies. Below are the metrics from the validation set.

### Training Metrics
![Training Curves](assets/images/results_graph.png)
*Training graphs showing the convergence of Box Loss and increase in mAP over epochs.*

### Confusion Matrix & Precision
![Confusion Matrix](assets/images/confusion_matrix.png)

**Key Performance Indicators:**
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **mAP@0.5** | **0.9X** | Excellent detection rate at standard threshold. |
| **mAP@0.5:0.95** | **0.XX** | High localization accuracy. |
| **Inference Speed** | **~25ms** | Real-time processing capability (on NVIDIA GPU). |

---

## Technical Stack

**Data Science & AI**
* Python (Pandas, NumPy, OpenCV)
* PyTorch (CUDA supported)
* Ultralytics YOLOv8/v11

**Web Development**
* **Backend:** FastAPI, Pydantic, SQLAlchemy, Uvicorn
* **Frontend:** Vue.js 3, Nuxt, TailwindCSS, Axios
* **Database:** SQLite (Dev), PostgreSQL (Prod ready)

**DevOps & Tools**
* Docker & Docker Compose
* Git / GitHub
* NVIDIA Container Toolkit

---

## Key Features Highlights
1.  **Human-in-the-loop:** The AI provides the initial count, but users can refine the results. These "corrected" annotations are saved and can be used to re-train the model, creating a continuous learning loop.
2.  **Report Generation:** Automated PDF/CSV export containing image thumbnails, total counts, and confidence scores for lab records.
3.  **Adaptive Thresholding:** Users can adjust confidence thresholds dynamically on the frontend to filter out noise without re-running the model.

---

## Contact
**[Your Name]**
* 🎓 Student at Faculty of ICT, Mahidol University
* 📧 Email: [Your Email]
* 🔗 LinkedIn: [Your LinkedIn Profile]

---
*© 2025 [Your Name]. All Rights Reserved.*
