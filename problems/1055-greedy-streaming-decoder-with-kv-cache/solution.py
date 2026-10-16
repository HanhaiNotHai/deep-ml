import numpy as np


def greedy_stream(model: 'ToyLM', prompt_ids: list[int], max_new_tokens: int, eos_id: int) -> list:
    """
    Greedy streaming decoder backed by a KV cache.

    Args:
        model: object exposing prefill(prompt_ids) -> np.ndarray of logits
               and decode_step(token_id) -> np.ndarray of logits.
        prompt_ids: iterable of int token ids.
        max_new_tokens: maximum number of tokens to generate.
        eos_id: token id that terminates generation.

    Returns:
        List of generated token ids (excluding the prompt and EOS).
    """

    out = []
    logits = model.prefill(prompt_ids)
    for _ in range(max_new_tokens):
        tid = logits.argmax().item()
        if tid == eos_id:
            break
        out.append(tid)
        logits = model.decode_step(out[-1])
    return out
