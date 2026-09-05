import os
import hashlib
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class ArcFaceMatcher:
    """
    ArcFace Biometric Recognizer:
    - Checks for ONNX Runtime model at models/arcface.onnx
    - In dev mode / fallback, uses deterministic perceptual feature projection to 512 dimensions.
    """
    def __init__(self, model_path: str = "models/arcface.onnx"):
        self.model_path = model_path
        self.onnx_session = None
        self._init_model()

    def _init_model(self):
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.onnx_session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                print(f"[ArcFace] Model ONNX loaded from {self.model_path}")
            except Exception as e:
                print(f"[ArcFace] Warning loading ONNX model: {e}. Fallback to perceptual projection.")
        else:
            print(f"[ArcFace] ONNX model not found at '{self.model_path}'. Running deterministic perceptual embedding mode.")

    def extract_embedding(self, face_roi: np.ndarray) -> List[float]:
        """
        Extracts a 512-dimension unit-length embedding vector from a 112x112 face ROI.
        """
        if self.onnx_session is not None:
            try:
                # Preprocess for ArcFace: BGR -> RGB, normalized to [-1, 1]
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
                print(f"[ArcFace] Inference error: {e}, falling back to projection.")

        # Deterministic perceptual projection fallback (Zero dependency, guaranteed 512-d unit vector)
        resized = cv2.resize(face_roi, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 2D DCT feature reduction
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:16, :16].flatten() # 256 coefficients
        
        # Image content hash for unique seed identity
        img_hash = hashlib.sha256(gray.tobytes()).hexdigest()
        seed = int(img_hash[:8], 16)
        rng = np.random.default_rng(seed)
        random_proj = rng.standard_normal((256, 512))
        
        embedding = np.dot(dct_low, random_proj)
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

    def find_match(self, query_embedding: List[float], registry: List[Dict[str, Any]], threshold: float = 0.75) -> Tuple[Optional[Dict[str, Any]], float]:
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
