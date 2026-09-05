import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ArcFaceMatcher:
    """
    ArcFace Biometric Recognizer:
    - Loads official InsightFace MobileFaceNet ArcFace 512-D ONNX model natively via OpenCV DNN (cv2.dnn.readNetFromONNX)
      or ONNX Runtime if available.
    - Extracts 512-dimension unit-length embedding vectors.
    - Fallback: deterministic fixed-matrix perceptual projection (never per-frame noise seed).
    """
    def __init__(self, model_path: str = "models/arcface.onnx"):
        self.model_path = model_path
        self.dnn_net = None
        self.onnx_session = None

        # Fixed deterministic projection matrix for fallback mode
        # Initialized ONCE so that same face always produces identical embeddings
        rng = np.random.default_rng(20260905)
        raw_mat = rng.standard_normal((256, 512))
        self.fixed_proj = raw_mat / np.linalg.norm(raw_mat, axis=0, keepdims=True)

        self._init_model()

    def _init_model(self):
        if os.path.exists(self.model_path):
            # 1. Try OpenCV DNN (native, fast, no external onnxruntime dependency)
            try:
                self.dnn_net = cv2.dnn.readNetFromONNX(self.model_path)
                self.dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"[ArcFace] Model successfully loaded via OpenCV DNN from {self.model_path}")
                return
            except Exception as e_dnn:
                print(f"[ArcFace] OpenCV DNN load notice: {e_dnn}. Trying ONNX Runtime...")

            # 2. Try ONNX Runtime (if installed)
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                print(f"[ArcFace] Model loaded via ONNX Runtime from {self.model_path}")
                return
            except Exception as e_ort:
                print(f"[ArcFace] ONNX Runtime load notice: {e_ort}.")

        print(f"[ArcFace] ONNX model not loaded. Running deterministic perceptual embedding fallback mode.")

    def extract_embedding(self, face_roi: np.ndarray) -> List[float]:
        """
        Extracts a 512-dimension unit-length embedding vector from a face ROI.
        """
        # 1. Native OpenCV DNN inference with real InsightFace ArcFace weights
        if self.dnn_net is not None:
            try:
                # InsightFace ArcFace standard input: 112x112, BGR->RGB, normalized to [-1, 1]
                blob = cv2.dnn.blobFromImage(
                    face_roi,
                    scalefactor=1.0 / 127.5,
                    size=(112, 112),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True,
                    crop=False
                )
                self.dnn_net.setInput(blob)
                out = self.dnn_net.forward()
                embedding = out[0]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.tolist()
            except Exception as e:
                print(f"[ArcFace] OpenCV DNN inference error: {e}")

        # 2. ONNX Runtime inference
        if self.onnx_session is not None:
            try:
                resized = cv2.resize(face_roi, (112, 112))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
                normalized = (rgb - 127.5) / 128.0
                blob = np.transpose(normalized, (2, 0, 1))
                blob = np.expand_dims(blob, axis=0)

                input_name = self.onnx_session.get_inputs()[0].name
                outputs = self.onnx_session.run(None, {input_name: blob})
                embedding = outputs[0][0]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding.tolist()
            except Exception as e:
                print(f"[ArcFace] ONNX Runtime inference error: {e}")

        # 3. Deterministic perceptual projection fallback (Constant matrix, never per-image random seed)
        resized = cv2.resize(face_roi, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 2D DCT feature reduction
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:16, :16].flatten() # 256 coefficients
        
        embedding = np.dot(dct_low, self.fixed_proj)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = np.ones(512) / np.sqrt(512)

        return embedding.tolist()

    @staticmethod
    def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
        if len(vector_a) != len(vector_b) or len(vector_a) == 0:
            return 0.0
        va = np.array(vector_a, dtype=np.float32)
        vb = np.array(vector_b, dtype=np.float32)
        dot = np.dot(va, vb)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def find_match(self, query_embedding: List[float], registry: List[Dict[str, Any]], threshold: float = 0.65) -> Tuple[Optional[Dict[str, Any]], float]:
        best_emp = None
        best_sim = 0.0
        for emp in registry:
            references = emp.get("embeddings") or ([emp.get("embedding")] if emp.get("embedding") else [])
            for ref_vec in references:
                if not ref_vec:
                    continue
                sim = self.cosine_similarity(query_embedding, ref_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_emp = emp

        if best_sim >= threshold:
            return best_emp, round(best_sim, 4)
        return None, round(best_sim, 4)
