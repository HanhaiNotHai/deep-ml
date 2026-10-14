from itertools import pairwise


def bpe_encode(text: str, token_to_id: dict[str, int], merges: dict[tuple[int, int], int]):
    tokens = [token_to_id[t] for t in text]

    while True:
        best_pair = None
        best_id = None
        for pair in pairwise(tokens):
            if pair in merges:
                new_id = merges[pair]
                if best_id is None or new_id < best_id:
                    best_pair = pair
                    best_id = new_id
        if best_pair is None:
            break

        merged = []
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == best_pair:
                merged.append(best_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        if i == len(tokens) - 1:
            merged.append(tokens[i])

        tokens = merged

    return tokens
