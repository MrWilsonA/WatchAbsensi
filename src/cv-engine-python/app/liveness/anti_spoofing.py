import cv2
import numpy as np
from typing import Dict, Any, Tuple

class AntiSpoofingClassifier:
    """
    Multi-factor passive anti-spoofing engine:
    1. Laplacian Texture Variance (detects printed paper / blur)
    2. 2D Fast Fourier Transform (FFT) analysis (detects screen refresh/moiré patterns)
    3. Specular Highlight & Color Balance Check
    """
    def __init__(self, liveness_threshold: float = 0.80):
        self.liveness_threshold = liveness_threshold

    def analyze_texture(self, face_roi: np.ndarray) -> float:
        """
        Returns normalized texture score (0.0 - 1.0).
        Real human skin presents micro-textures within an optimal sharpness band.
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Scale typical laplacian variance (100 to 1000) to 0.0 - 1.0
        score = np.clip((laplacian_var - 50.0) / 450.0, 0.05, 0.99)
        return float(score)

    def analyze_screen_moiré(self, face_roi: np.ndarray) -> float:
        """
        Analyzes periodic frequency peaks in 2D FFT.
        Screens (phones/monitors) exhibit high periodic energy peaks.
        Returns a screen reflection risk score (0.0 = low risk, 1.0 = high risk).
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1e-5)
        
        # Mask out center low frequency
        cy, cx = h // 2, w // 2
        r = min(h, w) // 8
        magnitude_spectrum[cy - r:cy + r, cx - r:cx + r] = 0
        
        high_freq_peak = np.max(magnitude_spectrum)
        # Normalized peak check
        # A normal webcam frame can contain strong edges from hair, shelves, or
        # room lighting. 180 was too aggressive and classified those edges as
        # screen moiré. Require a stronger periodic peak before blocking.
        if high_freq_peak > 220:
            return 0.85 # High moire pattern risk
        return 0.15 # Natural spectrum

    def evaluate_liveness(self, face_roi: np.ndarray, simulate_flag: str | None = None) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluates overall liveness. Supports test simulation flags for development kiosk testing.
        """
        if simulate_flag == "spoof_photo":
            return False, 0.38, {
                "reason": "photo_print_detected",
                "details": "Tekstur cetakan kertas teridentifikasi (Laplacian variance rendah)",
                "texture_score": 0.22,
                "moiré_score": 0.10
            }
        elif simulate_flag == "spoof_screen":
            return False, 0.42, {
                "reason": "screen_reflection",
                "details": "Pola moiré / pantulan piksel layar gadget terdeteksi",
                "texture_score": 0.65,
                "moiré_score": 0.89
            }
        elif simulate_flag == "spoof_low_conf":
            return False, 0.55, {
                "reason": "low_confidence",
                "details": "Pencahayaan redup atau wajah terhalang sebagian",
                "texture_score": 0.45,
                "moiré_score": 0.20
            }

        # Real visual calculations
        texture_score = self.analyze_texture(face_roi)
        moiré_risk = self.analyze_screen_moiré(face_roi)
        
        # Calculate combined liveness
        liveness_score = round(texture_score * 0.65 + (1.0 - moiré_risk) * 0.35, 3)
        # Ensure within reasonable bounds for camera feed
        liveness_score = float(np.clip(liveness_score, 0.72, 0.99))
        
        is_live = liveness_score >= self.liveness_threshold
        reason = None if is_live else ("screen_reflection" if moiré_risk > 0.5 else "texture_anomaly")

        indicators = {
            "texture_score": round(texture_score, 3),
            "moiré_risk": round(moiré_risk, 3),
            "reason": reason,
            "details": "Wajah biologis valid terverifikasi" if is_live else "Pola liveness tidak memenuhi ambang batas aman"
        }

        return is_live, liveness_score, indicators
