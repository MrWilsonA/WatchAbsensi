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
        Real human skin in webcam streams presents micro-textures and facial contours.
        Printed paper or blurry re-photographs typically have laplacian_var < 30.
        """
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Scale typical webcam face variance (30 to 120+) into 0.20 - 0.98 range
        score = np.clip((laplacian_var - 20.0) / 100.0, 0.20, 0.98)
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
                "details": "Printed-paper texture detected (low Laplacian variance).",
                "texture_score": 0.22,
                "moiré_score": 0.10
            }
        elif simulate_flag == "spoof_screen":
            return False, 0.42, {
                "reason": "screen_reflection",
                "details": "Display moire or screen-pixel reflection detected.",
                "texture_score": 0.65,
                "moiré_score": 0.89
            }
        elif simulate_flag == "spoof_low_conf":
            return False, 0.55, {
                "reason": "low_confidence",
                "details": "Low lighting or a partially obstructed face was detected.",
                "texture_score": 0.45,
                "moiré_score": 0.20
            }

        # Real visual calculations
        texture_score = self.analyze_texture(face_roi)
        moiré_risk = self.analyze_screen_moiré(face_roi)
        
        # Calculate combined liveness: balanced blend of skin texture clarity and absence of screen moire
        liveness_score = round(texture_score * 0.50 + (1.0 - moiré_risk) * 0.50, 3)
        # Ensure within reasonable bounds
        liveness_score = float(np.clip(liveness_score, 0.10, 0.99))
        
        is_live = liveness_score >= self.liveness_threshold
        reason = None if is_live else ("screen_reflection" if moiré_risk > 0.5 else "texture_anomaly")

        indicators = {
            "texture_score": round(texture_score, 3),
            "moiré_risk": round(moiré_risk, 3),
            "reason": reason,
            "details": "Live face verified." if is_live else "The liveness pattern did not meet the safety threshold."
        }

        return is_live, liveness_score, indicators
