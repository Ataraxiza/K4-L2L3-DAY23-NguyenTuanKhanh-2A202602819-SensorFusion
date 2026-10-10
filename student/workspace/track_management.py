"""Track initialization, scoring, and deletion helpers.

Part H supplies lidar-driven existence decisions (docs/HUONG_DAN_KY_THUAT.md §2).
Use tracking parameters for the score window, thresholds, and covariance limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fusion_lab.workspace_support import get_tracking_params


def init_track_state_from_meas(meas: Any) -> dict[str, Any]:
    """Initialize track state, covariance, lifecycle state, and score from a measurement.

    Args:
        meas: Lidar measurement with ``z``, ``R``, ``sensor``.

    Returns:
        Dict with keys ``x``, ``P``, ``state``, ``score`` (matrices as ``np.matrix``).
    """
    params = get_tracking_params()
    position = np.asarray(meas.z, dtype=float).reshape(-1)[:3]
    transform = np.asarray(meas.sensor.sens_to_veh)
    pos_vehicle = transform[:3, :3] @ position + transform[:3, 3]
    x = np.asmatrix(np.r_[pos_vehicle, [0.0, 0.0, 0.0]]).T

    rotation = transform[:3, :3]
    pos_cov = np.asarray(meas.R)[:3, :3]
    pos_cov = rotation @ pos_cov @ rotation.T
    vel_cov = np.diag([params.sigma_p44**2, params.sigma_p55**2, params.sigma_p66**2])
    P = np.asmatrix(np.zeros((6, 6)))
    P[:3, :3] = np.asmatrix(pos_cov)
    P[3:, 3:] = np.asmatrix(vel_cov)
    return {"x": x, "P": P, "score": 1.0 / params.window, "state": "initialized"}


def update_track_score(track: dict[str, Any], associated: bool) -> dict[str, Any]:
    """Update existence once per lidar frame; camera passes never call this helper.

    A hit adds 1/window, capped at one; an in-FOV miss subtracts 1/window.
    Confirm above confirmed_threshold, and preserve confirmed state after misses.

    Args:
        track: Dict-like track with ``score``, ``state``.
        associated: True for a lidar hit; False for a lidar miss within the lidar FOV.

    Returns:
        Updated track dict.
    """
    params = get_tracking_params()
    step = 1.0 / params.window
    updated_score = track["score"] + (step if associated else -step)
    if associated:
        updated_score = min(updated_score, 1.0)
    else:
        updated_score = max(updated_score, 0.0)

    state = track["state"]
    if state == "confirmed":
        next_state = "confirmed"
    elif updated_score > params.confirmed_threshold:
        next_state = "confirmed"
    elif updated_score > 0.0:
        next_state = "tentative"
    else:
        next_state = "initialized"

    track["score"] = float(updated_score)
    track["state"] = next_state
    return track


def should_delete_track(track: dict[str, Any]) -> bool:
    """Return whether a lidar lifecycle pass should remove this track.

    Delete if either horizontal variance exceeds max_P, or if a confirmed
    track has score < delete_threshold, or an unconfirmed track has score <= 0.
    Camera passes never trigger deletion.

    Args:
        track: Dict with ``score``, ``state``, ``P``.

    Returns:
        True if track should be removed.
    """
    params = get_tracking_params()
    P = np.asarray(track["P"], dtype=float)
    horizontal_var = float(max(P[0, 0], P[1, 1]))
    if horizontal_var > params.max_P:
        return True

    score = float(track["score"])
    state = track["state"]
    if state == "confirmed":
        return score < params.delete_threshold
    return score <= 0.0
