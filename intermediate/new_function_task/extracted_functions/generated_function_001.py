"""
Generated function extracted from New Function Task C dry-run.

This file was produced by:
scripts/new_function_task_C_dryrun_generate_one_function.py

This is a mock-LLM dry-run function, not a real LLM-generated final function.
"""


import numpy as np

class GeneratedDryRunLandscape:
    def __init__(self, dim=5):
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def f(self, x):
        x = np.asarray(x, dtype=float)

        if x.shape[0] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.shape[0]}")

        shifted = x - 1.25

        quadratic_basin = np.sum(shifted ** 2)

        multimodal_ripples = 0.15 * np.sum(np.sin(3.0 * x) ** 2)

        weak_interaction = (
            0.03 * np.sum(np.sin(x[:-1] * x[1:]))
            if self.dim > 1
            else 0.0
        )

        return float(quadratic_basin + multimodal_ripples + weak_interaction)
