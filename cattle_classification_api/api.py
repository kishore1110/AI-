from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import numpy as np
import io
import os
import gc
import traceback
import time

# ---------- Prometheus Monitoring ----------
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge

# Total /predict requests received
predict_requests_total = Counter(
    "cattle_predict_requests_total",
    "Total number of /predict requests received"
)
# Successful breed predictions
predict_success_total = Counter(
    "cattle_predict_success_total",
    "Total successful cattle breed predictions"
)
# Requests where no cattle was found in the image
predict_no_cattle_total = Counter(
    "cattle_predict_no_cattle_total",
    "Requests where no cattle was detected in the image"
)
# Requests that raised an exception
predict_errors_total = Counter(
    "cattle_predict_errors_total",
    "Total failed /predict requests due to an exception"
)
# End-to-end latency histogram for /predict
predict_latency_seconds = Histogram(
    "cattle_predict_latency_seconds",
    "End-to-end latency of the /predict endpoint in seconds",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
)
# Per-breed prediction counter (labelled)
breed_predicted_total = Counter(
    "cattle_breed_predicted_total",
    "Number of predictions per cattle breed",
    ["breed"]
)
# Gauge: 1 when both models are loaded and ready, 0 otherwise
models_loaded_gauge = Gauge(
    "cattle_models_loaded",
    "Whether ML models (classifier + detector) are loaded and ready (1=ready, 0=not ready)"
)
# ------------------------------------------

# Global model handles
classifier = None
detector = None
transform = None

# FIX: device must be global so the /predict endpoint can access it.
# Previously it was only defined inside lifespan(), causing a NameError on every request.
device = torch.device("cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, release on shutdown."""
    global classifier, detector, transform

    print("===== Application Startup at", __import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), "=====", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"NumPy version: {np.__version__}", flush=True)

    # --- Load EfficientNet classifier ---
    print("Loading EfficientNet classifier...", flush=True)
    _classifier = models.efficientnet_b0(weights=None)
    _classifier.classifier[1] = nn.Linear(_classifier.classifier[1].in_features, 41)
    state_dict = torch.load("efficientnet_70.pth", map_location=device)
    _classifier.load_state_dict(state_dict, strict=False)
    _classifier = _classifier.to(device).eval()
    classifier = _classifier
    # Free the raw state_dict immediately to reduce peak RAM
    del state_dict
    gc.collect()
    print("EfficientNet loaded.", flush=True)

    # --- Load YOLO detector ---
    print("Loading YOLO detector...", flush=True)
    if os.path.exists("yolov8n.pt"):
        detector = YOLO("yolov8n.pt")
        gc.collect()
        print("YOLO loaded.", flush=True)
    else:
        print("WARNING: yolov8n.pt not found. Detection will be disabled.", flush=True)

    # --- Build image transform pipeline ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("All models ready. Server is accepting requests.", flush=True)
    models_loaded_gauge.set(1)   # Signal to Prometheus that models are up
    yield
    models_loaded_gauge.set(0)   # Signal to Prometheus that models are going down
    print("Shutting down.", flush=True)

app = FastAPI(
    lifespan=lifespan,
    title="Cattle Classification API",
    description="YOLOv8 + EfficientNet-B0 cattle breed classifier with Prometheus monitoring.",
    version="1.0.0"
)

# Expose /metrics endpoint for Prometheus scraping.
# This must be called BEFORE adding other middleware.
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    inprogress_labels=True,
).instrument(app).expose(app, include_in_schema=False, tags=["monitoring"])

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

num_classes = 41

class_names = {
    0: "Unknown_0",    1: "Amritmahal",    2: "Unknown_2",     3: "Unknown_3",
    4: "Bargur",       5: "Banni",         6: "Purnea",        7: "Dangi",
    8: "Binjharpuri",  9: "Gir",          10: "Unknown_10",   11: "Shweta Kapila",
   12: "Ghumusari",   13: "Holstein",     14: "Unknown_14",   15: "Lakhimi",
   16: "Unknown_16",  17: "Kankrej",      18: "Malnad",       19: "Kherigarh",
   20: "Unknown_20",  21: "Khillar",      22: "Dagri",        23: "Badri",
   24: "Unknown_24",  25: "Unknown_25",   26: "Bachaur",      27: "Chilika",
   28: "Siri",        29: "Nimari",       30: "Unknown_30",   31: "Kangayam",
   32: "Khariar",     33: "Unknown_33",   34: "Poda Thurpu",  35: "Sahiwal",
   36: "Unknown_36",  37: "Unknown_37",   38: "Murrah Buffalo", 39: "Umblachery",
   40: "Motu"
}

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Cattle Classification API is running.", "docs": "/docs"}

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {
        "status": "ok",
        "classifier_loaded": classifier is not None,
        "detector_loaded": detector is not None
    }

@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not loaded. Check container logs.")
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not loaded. Check container logs.")

    # --- Track every incoming request ---
    predict_requests_total.inc()
    request_start = time.time()

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- YOLO detection ---
        results = detector(image, imgsz=640, conf=0.3, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            predict_no_cattle_total.inc()
            predict_latency_seconds.observe(time.time() - request_start)
            return {"success": False, "message": "No objects detected in the image."}

        cow_detected = False
        best_breed = None
        best_conf = 0.0

        for i in range(len(results.boxes)):
            cls_id = int(results.boxes.cls[i])
            conf_val = float(results.boxes.conf[i])

            # COCO class 19 = cow
            if cls_id == 19 and conf_val >= 0.3:
                cow_detected = True

                box = results.boxes.xyxy[i]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                x1, y1, x2, y2 = map(int, box)

                cropped = image.crop((x1, y1, x2, y2))
                input_tensor = transform(cropped).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = classifier(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    pred_conf, pred = torch.max(probabilities, 0)

                    class_idx = pred.item()
                    breed_name = class_names.get(class_idx, f"Unknown_{class_idx}")

                    if float(pred_conf) > best_conf:
                        best_conf = float(pred_conf)
                        best_breed = breed_name

        elapsed = time.time() - request_start
        predict_latency_seconds.observe(elapsed)

        if cow_detected and best_breed:
            predict_success_total.inc()
            breed_predicted_total.labels(breed=best_breed).inc()  # Track per-breed
            return {
                "success": True,
                "breed": best_breed,
                "confidence": round(best_conf, 4)
            }
        else:
            predict_no_cattle_total.inc()
            return {"success": False, "message": "No cattle detected in the image."}

    except Exception as e:
        predict_errors_total.inc()
        predict_latency_seconds.observe(time.time() - request_start)
        # Print full traceback to container logs so the error is visible in HF Spaces
        print("ERROR in /predict:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
