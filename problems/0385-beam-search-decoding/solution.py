import numpy as np


def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.

    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return

    Returns:
        List of (sequence, score) tuples sorted by score descending
    """

    beams = [([], 0)]
    for probs in log_probs:
        beams = [
            (seq + [token], score + float(probe))
            for seq, score in beams
            for token, probe in enumerate(probs)
        ]
        beams = sorted(beams, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[:top_k]
