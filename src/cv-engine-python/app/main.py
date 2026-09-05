import os
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.detector.face_detector import FaceDetector
from app.liveness.anti_spoofing import AntiSpoofingClassifier
from app.recognizer.arcface_matcher import ArcFaceMatcher
from app.analytics.workforce_analytics import WorkforceAnalyticsEngine

app = FastAPI(
    title="WatchAbsensi CV & Deep Learning Inference Engine",
    description="Live Biometric Pipeline: Face Detection, Anti-Spoofing Liveness, ArcFace 512-d Embedding, and Workforce Analytics.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize CV submodules
detector = FaceDetector()
anti_spoofing = AntiSpoofingClassifier(liveness_threshold=0.75)
recognizer = ArcFaceMatcher(model_path="models/arcface.onnx")

# In-memory stores
attendance_records: List[Dict[str, Any]] = []
face_registry: Dict[str, Dict[str, Any]] = {}
spoof_logs: List[Dict[str, Any]] = [
    {
        "id": "SP-001",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device_id": "KIOSK-MAIN-01",
        "liveness": 0.38,
        "confidence": 0.84,
        "reason": "photo_print_detected",
        "details": "Printed-paper texture detected (low Laplacian variance)",
        "candidate_id": "EMP-003"
    }
]

analytics_engine = WorkforceAnalyticsEngine(attendance_records, spoof_logs)

class AttendanceRequest(BaseModel):
    employee_id: str = Field(..., min_length=2)
    mode: str = Field(default="check-in", pattern="^(check-in|check-out)$")
    confidence: float = Field(default=0.98, ge=0, le=1)
    liveness: float = Field(default=0.99, ge=0, le=1)
    device_id: Optional[str] = "KIOSK-MAIN-01"

class SpoofReport(BaseModel):
    device_id: str = "KIOSK-MAIN-01"
    liveness: float
    confidence: float
    reason: str
    details: Optional[str] = None
    candidate_employee_id: Optional[str] = None

class FaceEnrollmentRequest(BaseModel):
    employee_id: str = Field(..., min_length=2)
    employee_name: str = Field(..., min_length=2)
    embeddings: List[List[float]] = Field(..., min_length=1, max_length=5)

class FaceRecognitionRequest(BaseModel):
    embedding: List[float] = Field(..., min_length=512, max_length=512)
    threshold: float = Field(default=0.75, ge=0.5, le=0.99)

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "watchabsensi-cv-engine",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "onnx_runtime_active": recognizer.onnx_session is not None
    }

@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "subsystems": {
            "face_detector": "operational",
            "anti_spoofing": "operational",
            "arcface_recognizer": "operational",
            "workforce_analytics": "operational"
        }
    }

@app.post("/v1/inference")
async def inference(
    frame: UploadFile = File(...),
    simulate_flag: Optional[str] = Query(None, description="Test simulation flag: 'spoof_photo', 'spoof_screen', 'spoof_low_conf'"),
    purpose: Optional[str] = Query(None, description="Use 'enrollment' for the multi-angle enrollment threshold")
):
    """
    Core CV Pipeline:
    1. Decodes frame image.
    2. Detects facial bounding box and crops face ROI.
    3. Analyzes anti-spoofing liveness (texture, moiré, reflections).
    4. Extracts 512-dimension biometric embedding vector.
    """
    image_bytes = await frame.read()
    digest = hashlib.sha256(image_bytes).hexdigest()[:12]

    try:
        face_found, boxes, full_img = detector.detect_faces(image_bytes)
    except Exception as e:
        return {
            "face_detected": False,
            "error": f"Unable to decode frame: {str(e)}",
            "liveness": 0.0,
            "faces": 0
        }

    if not face_found or len(boxes) == 0:
        return {
            "face_detected": False,
            "liveness": 0.0,
            "faces": 0,
            "message": "No face was detected in the frame."
        }

    primary_box = boxes[0]
    face_roi = detector.crop_face(full_img, primary_box)

    # Liveness check
    is_live, liveness_score, indicators = anti_spoofing.evaluate_liveness(face_roi, simulate_flag=simulate_flag)
    # 0.75 is calibrated for ordinary webcam sharpness; simulated print/screen
    # attacks remain well below this score and are still rejected.
    required_liveness = 0.75
    is_live = liveness_score >= required_liveness
    indicators["required_score"] = required_liveness
    indicators["details"] = "Live face verified." if is_live else f"Liveness score {liveness_score:.0%} is below the required {required_liveness:.0%}. Improve lighting and keep your full face inside the guide."

    # ArcFace embedding extraction (512-dim unit vector)
    embedding = recognizer.extract_embedding(face_roi)

    return {
        "face_detected": True,
        "faces": len(boxes),
        "bounding_box": primary_box,
        "is_live": is_live,
        "liveness": liveness_score,
        "indicators": indicators,
        "embedding_dimensions": len(embedding),
        "embedding": embedding,
        "embedding_id": digest,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/v1/faces/enroll")
def enroll_face(req: FaceEnrollmentRequest):
    if any(len(vector) != 512 for vector in req.embeddings):
        return {"success": False, "status": "invalid_embedding", "message": "Every face sample must contain a 512-dimensional vector."}
    for existing_id, existing in face_registry.items():
        if existing_id == req.employee_id:
            continue
        for incoming in req.embeddings:
            for registered in existing["embeddings"]:
                similarity = recognizer.cosine_similarity(incoming, registered)
                if similarity >= 0.92:
                    return {"success": False, "status": "already_registered", "message": f"This face is already registered as {existing['employee_name']}.", "existing_employee_id": existing_id, "existing_employee_name": existing["employee_name"], "similarity": round(similarity, 4)}
    face_registry[req.employee_id] = {"employee_id": req.employee_id, "employee_name": req.employee_name, "embeddings": req.embeddings, "enrolled_at": datetime.now(timezone.utc).isoformat()}
    return {"success": True, "status": "enrolled", "employee_id": req.employee_id, "samples": len(req.embeddings)}

@app.get("/v1/faces/registry")
def face_registry_status():
    return [{"employee_id": item["employee_id"], "employee_name": item["employee_name"], "samples": len(item["embeddings"]), "enrolled_at": item["enrolled_at"]} for item in face_registry.values()]

@app.post("/v1/faces/recognize")
def recognize_face(req: FaceRecognitionRequest):
    match, similarity = recognizer.find_match(req.embedding, list(face_registry.values()), req.threshold)
    if match is None:
        return {"matched": False, "similarity": similarity, "message": "This face is not registered."}
    return {"matched": True, "similarity": similarity, "employee_id": match["employee_id"], "employee_name": match["employee_name"]}

@app.post("/v1/liveness/verify")
async def verify_liveness(
    frame: UploadFile = File(...),
    simulate_flag: Optional[str] = Query(None)
):
    image_bytes = await frame.read()
    face_found, boxes, full_img = detector.detect_faces(image_bytes)
    if not face_found:
        return {"success": False, "is_live": False, "reason": "no_face_detected"}
    
    face_roi = detector.crop_face(full_img, boxes[0])
    is_live, score, indicators = anti_spoofing.evaluate_liveness(face_roi, simulate_flag=simulate_flag)
    return {
        "success": True,
        "is_live": is_live,
        "liveness_score": score,
        "indicators": indicators
    }

@app.post("/v1/attendance")
def record_attendance(req: AttendanceRequest):
    """
    Biometric attendance verification gate.
    Rejects spoof attempts or low confidence inferences and logs them.
    """
    if req.liveness < 0.75 or req.confidence < 0.75:
        spoof_item = {
            "id": f"SP-{len(spoof_logs)+1:03d}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "device_id": req.device_id,
            "liveness": req.liveness,
            "confidence": req.confidence,
            "reason": "liveness_failed" if req.liveness < 0.75 else "low_confidence",
            "candidate_id": req.employee_id
        }
        spoof_logs.append(spoof_item)
        return {
            "success": False,
            "status": "rejected",
            "reason": "Biometric verification failed: liveness or confidence is below the safety threshold.",
            "spoof_log": spoof_item
        }

    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "id": len(attendance_records) + 1,
        "employee_id": req.employee_id,
        "mode": req.mode,
        "recorded_at": now_iso,
        "confidence": req.confidence,
        "liveness": req.liveness,
        "device_id": req.device_id,
        "status": "OnTime"
    }
    attendance_records.append(record)

    return {
        "success": True,
        "status": "recorded",
        **record
    }

@app.get("/v1/attendance")
def attendance_history(limit: int = 50):
    clamped = min(max(limit, 1), 500)
    return attendance_records[-clamped:][::-1]

@app.get("/v1/attendance/spoof-logs")
def get_spoof_logs(limit: int = 50):
    return spoof_logs[-min(max(limit, 1), 200):][::-1]

@app.get("/v1/analytics/summary")
def summary():
    return analytics_engine.generate_summary(total_staff=150)

@app.get("/v1/analytics/hourly")
def hourly():
    return analytics_engine.generate_hourly_traffic()
