def replace_pair(sequences: list[list[int]], pair: tuple[int, int], new_id: int):
    rep_seqs = []
    for seq in sequences:
        rep_seqs.append([])
        i = 0
        while i < len(seq) - 1:
            if (seq[i], seq[i + 1]) == pair:
                rep_seqs[-1].append(new_id)
                i += 2
            else:
                rep_seqs[-1].append(seq[i])
                i += 1
        if i == len(seq) - 1:
            rep_seqs[-1].append(seq[i])
    return rep_seqs
