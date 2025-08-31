"""Filtering utilities for trend scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KalmanFilter1D:
    """A lightweight 1D Kalman filter.

    Parameters
    ----------
    process_noise : float
        Variance of the process noise (Q).
    measurement_noise : float
        Variance of the measurement noise (R).
    initial_estimate : float, optional
        Starting estimate for the state.
    initial_error : float, optional
        Initial error covariance (P).
    """

    process_noise: float = 1e-5
    measurement_noise: float = 1e-2
    initial_estimate: float = 0.0
    initial_error: float = 1.0

    def __post_init__(self) -> None:
        self.x = float(self.initial_estimate)
        self.p = float(self.initial_error)

    def update(self, measurement: float) -> float:
        """Ingest a new measurement and return the updated state estimate."""
        # Prediction step: project the error covariance ahead
        self.p += self.process_noise
        # Kalman gain
        k = self.p / (self.p + self.measurement_noise)
        # Update estimate with measurement
        self.x += k * (measurement - self.x)
        # Update error covariance
        self.p = (1 - k) * self.p
        return self.x

    @property
    def state(self) -> float:
        """Return the current state estimate."""
        return self.x

__all__ = ["KalmanFilter1D"]
