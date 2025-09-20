from __future__ import annotations

from typing import Dict

import numpy as np


def physical_violations(M: np.ndarray, qv: np.ndarray, qc: np.ndarray, qr: np.ndarray, *, eps_mass_rel: float = 1e-12) -> Dict[str, int]:
    eps_m = float(eps_mass_rel) * float(np.mean(M))
    return {
        "M_lt_eps": int(np.count_nonzero(M <= eps_m)),
        "qv_lt_0": int(np.count_nonzero(qv < -1e-16)),
        "qc_lt_0": int(np.count_nonzero(qc < -1e-16)),
        "qr_lt_0": int(np.count_nonzero(qr < -1e-16)),
    }