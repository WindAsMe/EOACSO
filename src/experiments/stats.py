"""Statistical significance tests for comparing algorithm variants across
independent runs: paired Wilcoxon signed-rank and Friedman."""

import numpy as np
from scipy import stats


def wilcoxon_signed_rank(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return {"statistic": None, "p_value": 1.0}
    statistic, p_value = stats.wilcoxon(a, b)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def friedman_test(*groups):
    statistic, p_value = stats.friedmanchisquare(*groups)
    return {"statistic": float(statistic), "p_value": float(p_value)}


def pairwise_wilcoxon(results_by_variant, reference):
    """results_by_variant: dict[name] -> array of per-run scores (same seeds/order).
    Returns Wilcoxon p-values comparing `reference` against every other variant."""
    ref = results_by_variant[reference]
    return {
        name: wilcoxon_signed_rank(ref, scores)
        for name, scores in results_by_variant.items()
        if name != reference
    }
