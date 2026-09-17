# SPDX-License-Identifier: Apache-2.0
"""Index arithmetic for the per-KV-group working set.

Pages hold one KV group and interleave by group, so a request's flat block list is

    flat block index = position * groups + group

and group g's block table is bt[:, g::groups]. Rows given to FlashAttention are
group-major: row = g * rows + i.

Slots within a group's frame (one contiguous range, so FlashAttention needs only a
length):

    [0, sel)             selected 16-token blocks (sel = 16 * blocks kept)
    [sel, sel + lf)      recency floor as of the last refresh
    [sel + lf, ...)      tokens decoded since that refresh

Token j (j >= floor start) therefore sits at slot sel + (j - floor_start).

No vLLM imports: tested directly in fork_tests/test_layout.py.
"""

import torch


def group_block_table(block_table: torch.Tensor, groups: int) -> torch.Tensor:
    """[rows, blocks] interleaved by group -> [groups * rows, blocks // groups], group-major rows."""
    rows, blocks = block_table.shape
    per_group = blocks // groups
    return (block_table[:, : per_group * groups].view(rows, per_group, groups)
            .permute(2, 0, 1).reshape(groups * rows, per_group).contiguous())


def row_of(group: int, row: int, rows: int) -> int:
    """Row index of (request row, KV group) in the group-major ordering."""
    return group * rows + row


def slot_pages(block_table_rows: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor,
               block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical page and offset of each (row, slot)."""
    pages = block_table_rows[rows, slots // block_size].to(torch.int64)
    return pages, slots % block_size


def token_slot(sel: int, floor_start: int, token: int) -> int:
    """Where token `token` sits in a group's frame, given that group's selected length."""
    return sel + (token - floor_start)


def decode_lengths_and_slots(sel: torch.Tensor, floor_len: torch.Tensor, tail: torch.Tensor,
                             no_slot_rows: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per layer and row: KV length of each group's frame, and the slot this step's token
    goes to (the frame's last slot).

    sel: [L, rows, groups] selected tokens; floor_len/tail: [rows] the floor length and the
    number of tokens decoded since that refresh, both of the frame the token is written into
    (for a refreshing row that is still the previous frame; its floor moves afterwards).
    Rows in no_slot_rows get -1: their first refresh takes the floor from the CPU store.
    """
    lens = sel + (floor_len + tail)[None, :, None]
    slots = lens - 1
    for r in no_slot_rows:
        slots[:, r, :] = -1
    return lens, slots


def floor_shift(sel_old: int, sel_new: int, prev_floor_start: int, floor_start: int,
                length: int) -> tuple[int, int, int]:
    """Moving the floor after a refresh: the tokens it now covers are already on the GPU,
    laid out from sel_old by the previous frame. Returns (src start, dst start, length)."""
    return sel_old + (floor_start - prev_floor_start), sel_new, length


def refresh_slots(resident_row: torch.Tensor, new_ids: torch.Tensor,
                  count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Which selected blocks a refresh must fetch, and which slots they go to.

    resident_row [K] holds the block id at each selected slot of one KV group, -1 for none;
    it is updated in place. new_ids[:count] is this refresh's selection for that group.

    Only slots below `count` count as resident: the frame is [0, count) blocks long, so a
    block sitting past it is not attended and has to come back from the CPU store. With
    that, need and free are the same length -- both are count - |old & new| -- so every
    dropped slot receives exactly one fetched block.
    """
    new_ids = new_ids.to(resident_row.dtype)
    resident_row[count:] = -1
    new, valid = new_ids[:count], resident_row[:count]
    need = new[~torch.isin(new, valid)]
    free = (~torch.isin(valid, new)).nonzero(as_tuple=True)[0]
    assert need.numel() == free.numel(), f"cascade: refresh slots {need.numel()} != {free.numel()}"
    resident_row[free] = need
    return need, free


def shift_indices(src0: list[int], dst0: list[int], lengths: list[int],
                  device) -> tuple[torch.Tensor, torch.Tensor]:
    """Source and destination slots for a batch of floor shifts, one row per (request, group).

    Every row moves `lengths[i]` slots from src0[i] to dst0[i], but the rows have different
    lengths, so they are padded to the widest and the padding entries are made to copy a slot
    ONTO ITSELF -- a no-op. That keeps the whole batch in one gather and one scatter;
    compacting the index list instead would need the lengths on the host, i.e. a sync.

    Returns (src, dst), each [rows, max(lengths)].
    """
    width = max(lengths)
    base = torch.arange(width, device=device)[None, :]
    dst = torch.tensor(dst0, device=device)[:, None] + base
    valid = base < torch.tensor(lengths, device=device)[:, None]
    src = torch.where(valid, torch.tensor(src0, device=device)[:, None] + base, dst)
    return src, dst
