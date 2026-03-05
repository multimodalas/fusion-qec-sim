import numpy as np


def apply_geometry_postprocessing(llr, structural):
    """
    Apply deterministic geometry field post-processing.

    Steps:
    1. optional variance normalization
    2. optional geometry strength scaling

    Only activates when geometry interventions are enabled.
    """

    geometry_active = structural.centered_field or structural.pseudo_prior
    if not geometry_active:
        return llr

    llr_arr = np.asarray(llr, dtype=np.float64).copy()

    # --- normalization ---
    if getattr(structural, "normalize_geometry", False):
        std = np.std(llr_arr)
        eps = 1e-9

        if std > eps:
            llr_arr = llr_arr / std
        # else: leave unchanged to avoid exploding values

    # --- strength scaling ---
    strength = getattr(structural, "geometry_strength", 1.0)

    if strength != 1.0:
        llr_arr = strength * llr_arr

    return llr_arr
