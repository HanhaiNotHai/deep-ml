import heapq
from collections import defaultdict

import numpy as np


class VirtualKVCacheManager:

    def __init__(self, num_physical_blocks: int, block_size: int, hidden_dim: int):
        """
        Initialize the virtual memory KV cache manager.

        Args:
            num_physical_blocks: Total number of physical memory blocks
            block_size: Number of token slots per block
            hidden_dim: Dimension of each key/value vector
        """

        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.keys = np.zeros((num_physical_blocks, block_size, hidden_dim))
        self.values = np.zeros_like(self.keys)

        self.free_blocks = list(range(num_physical_blocks))
        heapq.heapify(self.free_blocks)

        self.page_tables: defaultdict[int, list[int]] = defaultdict(list)

    def allocate_block(self, seq_id: int):
        """Allocate a new physical block for seq_id. Return logical index or -1."""

        if not self.free_blocks:
            return -1

        physical_block = heapq.heappop(self.free_blocks)
        page_table = self.page_tables[seq_id]
        page_table.append(physical_block)

        return len(page_table) - 1

    def write(self, seq_id: int, position: int, key: np.ndarray, value: np.ndarray):
        """Write KV pair at token position. Auto-allocates blocks. Return success."""

        logical_block, offset = divmod(position, self.block_size)
        page_table = self.page_tables[seq_id]

        needed = max(0, logical_block + 1 - len(page_table))
        if needed > len(self.free_blocks):
            return False

        for _ in range(needed):
            self.allocate_block(seq_id)

        physical_block = page_table[logical_block]
        self.keys[physical_block, offset] = key
        self.values[physical_block, offset] = value

        return True

    def read(self, seq_id: int, position: int):
        """Read KV pair at token position. Return (key, value) or (None, None)."""

        logical_block, offset = divmod(position, self.block_size)
        page_table = self.page_tables[seq_id]

        if logical_block >= len(page_table):
            return None, None

        physical_block = page_table[logical_block]
        return self.keys[physical_block, offset].copy(), self.values[physical_block, offset].copy()

    def free_sequence(self, seq_id: int):
        """Free all blocks for seq_id. Return number of blocks freed."""

        physical_blocks = self.page_tables[seq_id]

        for block in physical_blocks:
            self.keys[block].fill(0)
            self.values[block].fill(0)
            heapq.heappush(self.free_blocks, block)

        return len(physical_blocks)

    def get_num_free_blocks(self):
        """Return number of unallocated physical blocks."""

        return len(self.free_blocks)

    def get_page_table(self, seq_id: int):
        """Return list of physical block indices in logical order for seq_id."""

        return self.page_tables[seq_id]
