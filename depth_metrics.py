"""
hand_metrics.py
===============
Shared helper for calculating hand size metrics and smoothing them.
"""

import math
from typing import List, Optional
import numpy as np

class EMAFilter:
    """Simple Exponential Moving Average filter for scalars."""
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: Optional[float] = None

    def filter(self, new_val: float) -> float:
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


def get_rigid_palm_metric(landmarks: np.ndarray) -> float:
    """
    Computes a rigid palm size metric from 21 MediaPipe hand landmarks.
    Averages the distances between:
      - wrist (0) ↔ index MCP (5)
      - wrist (0) ↔ pinky MCP (17)
      - index MCP (5) ↔ pinky MCP (17)
    
    This forms a rigid triangle that does not change significantly when fingers
    curl or the hand tilts, making it a reliable depth proxy.
    
    Parameters
    ----------
    landmarks: A numpy array of shape (N, 2) where N is at least 18.
    
    Returns
    -------
    The average distance as a float.
    """
    if len(landmarks) < 18:
        return 0.1  # fallback to prevent zero division

    d1 = np.linalg.norm(landmarks[0] - landmarks[5])
    d2 = np.linalg.norm(landmarks[0] - landmarks[17])
    d3 = np.linalg.norm(landmarks[5] - landmarks[17])
    
    return float((d1 + d2 + d3) / 3.0)
