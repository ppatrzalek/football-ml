"""Per-player pitch-zone occupancy and spatial-dispersion features.

Consumes the tidy outfielder frame from
:func:`football_ml.formations.build_outfielder_frame`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BANDS = 8
N_LANES = 4
N_ZONES = N_BANDS * N_LANES


def zone_index(
    depth,
    lateral,
    pitch_length: float,
    pitch_width: float,
    *,
    n_bands: int = N_BANDS,
    n_lanes: int = N_LANES,
):
    """Map normalized (depth, lateral) coordinates to pitch-zone indices.

    Parameters
    ----------
    depth, lateral : float or array-like
        Attack-normalized coordinates in meters (0 = pitch center). ``depth``
        runs own-goal(-L/2) -> opponent-goal(+L/2); ``lateral`` runs
        own-left(-W/2) -> own-right(+W/2).
    pitch_length, pitch_width : float
        Pitch dimensions (L, W) in meters.
    n_bands, n_lanes : int
        Grid resolution along length and width.

    Returns
    -------
    tuple
        ``(band, lane, zone)``. Scalars in -> Python ints out; arrays in ->
        numpy int arrays out. ``zone = band * n_lanes + lane``. Out-of-pitch
        coordinates are clipped into the edge zones.
    """
    depth = np.asarray(depth, dtype=float)
    lateral = np.asarray(lateral, dtype=float)
    band = np.floor((depth + pitch_length / 2) / (pitch_length / n_bands))
    lane = np.floor((lateral + pitch_width / 2) / (pitch_width / n_lanes))
    band = np.clip(band, 0, n_bands - 1).astype(int)
    lane = np.clip(lane, 0, n_lanes - 1).astype(int)
    zone = band * n_lanes + lane
    if band.ndim == 0:
        return int(band), int(lane), int(zone)
    return band, lane, zone
