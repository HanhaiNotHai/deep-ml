def analyze_memory_fragmentation(block_status: list[int]):
    """
    Analyze memory fragmentation in a block-based memory pool.

    Args:
            block_status: List of ints where 1 = allocated block, 0 = free block

    Returns:
            dict with keys:
                    'utilization': float, fraction of allocated blocks
                    'num_free_fragments': int, count of contiguous free regions
                    'largest_free_fragment': int, size of largest contiguous free region
                    'fragmentation_ratio': float, measure of free memory scatter
    """

    n = len(block_status)
    alloc = 0
    num_free_fragments = 0
    largest_free_fragment = 0
    c = 0

    for x in block_status:
        if x == 0:
            c += 1
        else:
            alloc += 1
            if c > 0:
                largest_free_fragment = max(largest_free_fragment, c)
                c = 0
                num_free_fragments += 1
    if c > 0:
        largest_free_fragment = max(largest_free_fragment, c)
        num_free_fragments += 1

    return {
        'utilization': alloc / n,
        'num_free_fragments': num_free_fragments,
        'largest_free_fragment': largest_free_fragment,
        'fragmentation_ratio': 1 - largest_free_fragment / (n - alloc) if alloc < n else 0.0,
    }
