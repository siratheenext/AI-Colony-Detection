# AI Colony Detection (Colony-Count-YOLO)

Project summary
---------------
End-to-end web system for detecting and counting bacterial/cell colonies on petri-dish images. Work covers dataset collection and annotation, augmentation, training and fine-tuning YOLO models (YOLOv8 / YOLOv11), evaluation, an inference pipeline with a FastAPI backend, a Vue/Nuxt frontend for upload and interactive bounding-box editing, persistence to SQLite, CSV/PDF reporting, and deployment guidance (Docker, GPU).

Key features
------------
- Custom dataset: image collection, YOLO-format annotations, train/val split, and augmentations (crop, flip, CLAHE, histogram equalization, grayscale variants).
- Model training: transfer learning on YOLOv8 / YOLOv11; configurable imgsz, batch, epochs; saves best.pt and logs.
- Evaluation: automated validation with mAP@0.5 and mAP@0.5:0.95 and analysis of precision/recall and error cases.
- Inference API: FastAPI endpoints for image upload, model inference, result persistence and JSON responses.
- Frontend: Vue / Nuxt UI for upload, preview with overlayed boxes (canvas/SVG), manual edit and save.
- Reports & export: history, CSV and PDF export of counts and detection details.
- Deployment: Docker / docker-compose guidance; note for GPU inference using NVIDIA runtime.

Repository layout (example)
---------------------------
- colony_dataset/
  - images/train, images/val
  - labels/train, labels/val
- runs/                         # training & detect outputs (weights, logs, plots)
- scripts/
  - test_colony.py              # inference + evaluation example (Ultralytics API)
- backend/                      # FastAPI app (optional)
- frontend/                     # Vue / Nuxt app (optional)
- colony_data.yaml
- requirements.txt
- README.md

Dataset & annotation
--------------------
- Annotation format: YOLO per-line: class_id x_center y_center width height (normalized).
- Ensure label filenames share the same basename as images.
- Recommended workflow:
  1. Collect raw images.
  2. Annotate (annotation tool).
  3. Visual spot-check annotations for quality.
  4. Augment (as needed).
  5. Split into train/val.

Example colony_data.yaml
------------------------
```yaml
train: colony_dataset/images/train
val:   colony_dataset/images/val
nc: 1
names: ['colony']
```

Training (step-by-step)
-----------------------
1. Setup environment (Windows example):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# or at least:
pip install ultralytics opencv-python torch torchvision
```
2. Train example (Ultralytics CLI):
```powershell
yolo train model=yolov8n.pt data=colony_data.yaml imgsz=832 epochs=100 batch=16 project=runs/detect name=Colony-Count-train_V4
```
3. Recommendations:
- Use transfer learning (pretrained weights).
- Monitor validation metrics to avoid overfitting.
- Use augmentations suited to your imaging conditions.
- Save best.pt and training logs.

Evaluation & metrics
--------------------
- Use Ultralytics API: `metrics = model.val(data="colony_data.yaml", imgsz=832)`
- Primary metrics: mAP@0.5 (`metrics.box.map50`) and mAP@0.5:0.95 (`metrics.box.map`).
- Also analyze precision, recall, and inspect false positives/negatives visually.

Inference pipeline (backend)
----------------------------
High-level flow:
1. Frontend uploads image via multipart/form-data to API.
2. Backend saves image record in DB and a temporary file on disk.
3. YOLO model runs inference: `results = model(tmp_path, conf=threshold)`.
4. Parse results to JSON (boxes, confidences, class ids) and draw overlay preview.
5. Persist prediction (boxes JSON, count, model_version) to DB and return JSON to frontend.

Suggested API endpoints
- POST /predict — upload image -> returns { id, count, boxes, preview_url }
- GET /predictions/{id} — get stored prediction and metadata
- POST /predictions/{id}/edit — save user edits to boxes
- GET /export/csv?from=...&to=... — export CSV report

Database (suggested schema)
---------------------------
PoC: SQLite; production: PostgreSQL recommended.

- images (id, filename, path, uploaded_at)
- predictions (id, image_id, model_version, count, boxes_json, confidence_avg, user_edited, created_at)
- users (optional, for auth)

Frontend (UI)
-------------
- Upload page with image preview.
- Result page: original image + overlay bounding boxes (canvas or SVG), per-box confidence.
- Edit mode: add, move, resize, delete boxes; save the edited result to backend.
- History & reports page: list of past predictions with export buttons.

Post-processing & counting tips
-------------------------------
- Filter boxes below a min-area to reduce small false positives.
- Tune confidence threshold and NMS IoU based on colony density.
- For highly overlapping colonies, consider instance segmentation or density/counting models.

Deployment
----------
- Development: run FastAPI with Uvicorn and run frontend dev server.
- Production: Docker + docker-compose for backend, DB, frontend, reverse proxy.
- GPU inference: use PyTorch CUDA-compatible image and nvidia-container-toolkit / nvidia runtime.

Example docker-compose outline
------------------------------
```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./runs:/app/runs"]
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: example
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```

Usage examples
--------------
- Run inference script:
```powershell
python scripts\test_colony.py
```
- Start backend (dev):
```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- Train model:
```powershell
yolo train model=yolov8s.pt data=colony_data.yaml imgsz=832 epochs=100
```

Troubleshooting
---------------
- Model returns 0 boxes: verify model path, confidence threshold, and preprocessing.
- Low mAP: check annotation quality, increase data variety or augmentations, adjust training hyperparameters.
- Overlapping colonies cause miscounts: consider segmentation or specialized counting network.
- Docker GPU issues: confirm NVIDIA drivers and container runtime are installed.

Security & privacy
------------------
- Validate upload types and file size limits.
- Authenticate/authorize production endpoints.
- If data is sensitive, follow institutional data policies and de-identify as required.

Contributing
------------
- Fork → create feature branch → open PR with description, screenshots, and metrics.
- Add unit tests for new backend endpoints or processing steps.

License & contact
-----------------
- Choose and include a license (e.g., MIT) and provide contact information (email/GitHub).

Resume blurb (short)
--------------------
Built an end-to-end web system for bacterial colony counting: collected and annotated a custom dataset, trained and fine-tuned YOLOv8/YOLOv11, implemented a FastAPI backend and Vue/Nuxt frontend for upload and interactive bounding-box editing, persisted results to SQLite, and added CSV/PDF export. Deployed with Docker; includes training/evaluation scripts and an augmentation pipeline.

-- End of README --
```// filepath: c:\Users\sirat\Colony-Count-YOLO\README.md
# AI Colony Detection (Colony-Count-YOLO)

Project summary
---------------
End-to-end web system for detecting and counting bacterial/cell colonies on petri-dish images. Work covers dataset collection and annotation, augmentation, training and fine-tuning YOLO models (YOLOv8 / YOLOv11), evaluation, an inference pipeline with a FastAPI backend, a Vue/Nuxt frontend for upload and interactive bounding-box editing, persistence to SQLite, CSV/PDF reporting, and deployment guidance (Docker, GPU).

Key features
------------
- Custom dataset: image collection, YOLO-format annotations, train/val split, and augmentations (crop, flip, CLAHE, histogram equalization, grayscale variants).
- Model training: transfer learning on YOLOv8 / YOLOv11; configurable imgsz, batch, epochs; saves best.pt and logs.
- Evaluation: automated validation with mAP@0.5 and mAP@0.5:0.95 and analysis of precision/recall and error cases.
- Inference API: FastAPI endpoints for image upload, model inference, result persistence and JSON responses.
- Frontend: Vue / Nuxt UI for upload, preview with overlayed boxes (canvas/SVG), manual edit and save.
- Reports & export: history, CSV and PDF export of counts and detection details.
- Deployment: Docker / docker-compose guidance; note for GPU inference using NVIDIA runtime.

Repository layout (example)
---------------------------
- colony_dataset/
  - images/train, images/val
  - labels/train, labels/val
- runs/                         # training & detect outputs (weights, logs, plots)
- scripts/
  - test_colony.py              # inference + evaluation example (Ultralytics API)
- backend/                      # FastAPI app (optional)
- frontend/                     # Vue / Nuxt app (optional)
- colony_data.yaml
- requirements.txt
- README.md

Dataset & annotation
--------------------
- Annotation format: YOLO per-line: class_id x_center y_center width height (normalized).
- Ensure label filenames share the same basename as images.
- Recommended workflow:
  1. Collect raw images.
  2. Annotate (annotation tool).
  3. Visual spot-check annotations for quality.
  4. Augment (as needed).
  5. Split into train/val.

Example colony_data.yaml
------------------------
```yaml
train: colony_dataset/images/train
val:   colony_dataset/images/val
nc: 1
names: ['colony']
```

Training (step-by-step)
-----------------------
1. Setup environment (Windows example):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# or at least:
pip install ultralytics opencv-python torch torchvision
```
2. Train example (Ultralytics CLI):
```powershell
yolo train model=yolov8n.pt data=colony_data.yaml imgsz=832 epochs=100 batch=16 project=runs/detect name=Colony-Count-train_V4
```
3. Recommendations:
- Use transfer learning (pretrained weights).
- Monitor validation metrics to avoid overfitting.
- Use augmentations suited to your imaging conditions.
- Save best.pt and training logs.

Evaluation & metrics
--------------------
- Use Ultralytics API: `metrics = model.val(data="colony_data.yaml", imgsz=832)`
- Primary metrics: mAP@0.5 (`metrics.box.map50`) and mAP@0.5:0.95 (`metrics.box.map`).
- Also analyze precision, recall, and inspect false positives/negatives visually.

Inference pipeline (backend)
----------------------------
High-level flow:
1. Frontend uploads image via multipart/form-data to API.
2. Backend saves image record in DB and a temporary file on disk.
3. YOLO model runs inference: `results = model(tmp_path, conf=threshold)`.
4. Parse results to JSON (boxes, confidences, class ids) and draw overlay preview.
5. Persist prediction (boxes JSON, count, model_version) to DB and return JSON to frontend.

Suggested API endpoints
- POST /predict — upload image -> returns { id, count, boxes, preview_url }
- GET /predictions/{id} — get stored prediction and metadata
- POST /predictions/{id}/edit — save user edits to boxes
- GET /export/csv?from=...&to=... — export CSV report

Database (suggested schema)
---------------------------
PoC: SQLite; production: PostgreSQL recommended.

- images (id, filename, path, uploaded_at)
- predictions (id, image_id, model_version, count, boxes_json, confidence_avg, user_edited, created_at)
- users (optional, for auth)

Frontend (UI)
-------------
- Upload page with image preview.
- Result page: original image + overlay bounding boxes (canvas or SVG), per-box confidence.
- Edit mode: add, move, resize, delete boxes; save the edited result to backend.
- History & reports page: list of past predictions with export buttons.

Post-processing & counting tips
-------------------------------
- Filter boxes below a min-area to reduce small false positives.
- Tune confidence threshold and NMS IoU based on colony density.
- For highly overlapping colonies, consider instance segmentation or density/counting models.

Deployment
----------
- Development: run FastAPI with Uvicorn and run frontend dev server.
- Production: Docker + docker-compose for backend, DB, frontend, reverse proxy.
- GPU inference: use PyTorch CUDA-compatible image and nvidia-container-toolkit / nvidia runtime.

Example docker-compose outline
------------------------------
```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./runs:/app/runs"]
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: example
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
```

Usage examples
--------------
- Run inference script:
```powershell
python scripts\test_colony.py
```
- Start backend (dev):
```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
- Train model:
```powershell
yolo train model=yolov8s.pt data=colony_data.yaml imgsz=832 epochs=100
```

Troubleshooting
---------------
- Model returns 0 boxes: verify model path, confidence threshold, and preprocessing.
- Low mAP: check annotation quality, increase data variety or augmentations, adjust training hyperparameters.
- Overlapping colonies cause miscounts: consider segmentation or specialized counting network.
- Docker GPU issues: confirm NVIDIA drivers and container runtime are installed.

Security & privacy
------------------
- Validate upload types and file size limits.
- Authenticate/authorize production endpoints.
- If data is sensitive, follow institutional data policies and de-identify as required.

Contributing
------------
- Fork → create feature branch → open PR with description, screenshots, and metrics.
- Add unit tests for new backend endpoints or processing steps.

License & contact
-----------------
- Choose and include a license (e.g., MIT) and provide contact information (email/GitHub).

Resume blurb (short)
--------------------
Built an end-to-end web system for bacterial colony counting: collected and annotated a custom dataset, trained and fine-tuned YOLOv8/YOLOv11, implemented a FastAPI backend and Vue/Nuxt frontend for upload and interactive bounding-box editing, persisted results to SQLite, and added CSV/PDF export. Deployed with Docker; includes training/evaluation scripts and an augmentation pipeline.

-- End of README --
