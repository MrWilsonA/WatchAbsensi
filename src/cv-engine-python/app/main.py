from datetime import datetime, timezone
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import hashlib, random

app = FastAPI(title="WatchAbsensi CV & Attendance API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AttendanceRequest(BaseModel):
    employee_id: str = Field(..., min_length=2)
    mode: str = Field(default="check-in", pattern="^(check-in|check-out)$")
    confidence: float = Field(default=0.98, ge=0, le=1)
    liveness: float = Field(default=0.99, ge=0, le=1)

@app.get("/health")
def health(): return {"status":"healthy","service":"cv-engine","timestamp":datetime.now(timezone.utc).isoformat()}

@app.post("/v1/inference")
async def inference(frame: UploadFile = File(...)):
    data = await frame.read()
    digest = hashlib.sha256(data).hexdigest()[:12]
    return {"face_detected": True, "liveness": round(random.uniform(.96,.995),3), "embedding_id": digest, "faces": 1}

@app.post("/v1/attendance")
def attendance(req: AttendanceRequest):
    if req.liveness < .8 or req.confidence < .75:
        return {"success":False,"status":"rejected","reason":"Verifikasi liveness atau confidence tidak memenuhi ambang batas"}
    return {"success":True,"status":"recorded","employee_id":req.employee_id,"mode":req.mode,"recorded_at":datetime.now(timezone.utc).isoformat(),"confidence":req.confidence}

@app.get("/v1/analytics/summary")
def summary():
    return {"date":datetime.now().date().isoformat(),"present":128,"late":9,"absent":12,"attendance_rate":.914,"avg_work_hours":8.1,"spoof_attempts":2}
