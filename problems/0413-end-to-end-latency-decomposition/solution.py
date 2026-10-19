import numpy as np
from numpy.typing import NDArray


def decompose_latency(stage_latencies: dict[str, NDArray], percentiles: list[int]):
    """
    Decompose end-to-end inference latency into component stages.

    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])

    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """

    def get_percentiles(latency: NDArray):
        return {p: round(np.percentile(latency, p), 2) for p in percentiles}

    e2e = np.sum(list(stage_latencies.values()), axis=0)
    e2e_mean = e2e.mean()
    stage_means = {stage: latency.mean() for stage, latency in stage_latencies.items()}

    return {
        'e2e_mean': round(e2e_mean, 2),
        'e2e_percentiles': get_percentiles(e2e),
        'stage_stats': {
            stage: {
                'mean': round(stage_means[stage], 2),
                'std': round(latency.std(), 2),
                'percentiles': get_percentiles(latency),
            }
            for stage, latency in stage_latencies.items()
        },
        'bottleneck': max(stage_means, key=stage_means.get),
        'stage_pct': {
            stage: round(mean / e2e_mean * 100, 2) for stage, mean in stage_means.items()
        },
    }
