"""Best trade selection utilities."""

from __future__ import annotations

import pandas as pd


def select_best_trade(ranked: pd.DataFrame, min_edge: float, min_conf: float) -> pd.Series | None:
    """Return top trade row that satisfies edge score and confidence gates."""
    if ranked.empty:
        return None

    work = ranked.copy()
    work["edge_score"] = pd.to_numeric(work.get("edge_score"), errors="coerce")
    work["edge_score_confidence"] = pd.to_numeric(work.get("edge_score_confidence"), errors="coerce")
    gated = work[
        (work["edge_score"].notna())
        & (work["edge_score_confidence"].notna())
        & (work["edge_score"] >= min_edge)
        & (work["edge_score_confidence"] >= min_conf)
    ]
    if gated.empty:
        return None
    gated = gated.sort_values(["edge_score", "edge_score_confidence"], ascending=[False, False])
    return gated.iloc[0]
