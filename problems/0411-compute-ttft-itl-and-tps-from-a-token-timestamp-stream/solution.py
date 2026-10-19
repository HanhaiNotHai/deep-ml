def compute_inference_metrics(timestamps: list[float]):
    """
    Compute LLM inference performance metrics from token timestamps.

    Args:
            timestamps: List of floats where timestamps[0] is the request start time
                        and timestamps[1:] are the times when each output token was generated.

    Returns:
            Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
    """

    return {
        'ttft': timestamps[1] - timestamps[0],
        'tps': (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]),
        'itl': (
            (timestamps[-1] - timestamps[1]) / (len(timestamps) - 2)
            if len(timestamps) > 2
            else 0.0
        ),
    }
