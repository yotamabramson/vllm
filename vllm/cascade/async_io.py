# SPDX-License-Identifier: Apache-2.0
"""Transfers that do not block the decode step: the flush (GPU -> CPU store) and the
refresh fetch (CPU store -> GPU).

The problem this solves (measured, towards_70B_vllm/README.md "Full measurement set"):
a refresh step costs ~15.4 ms/step amortised and the flush another 6.8 ms of WALL time
against 0.1 ms of GPU time, on a ~37.5 ms step. Almost none of that is PCIe bytes
(~36 MB/step amortised against the ~500 MB/step the model itself reads); it is host
serialisation -- a `.cpu()` of the selected ids per layer, a host-side gather, and a
blocking host->device copy, with the step's own attention waiting behind all of it.

Both directions are moved onto ONE worker thread plus two side streams:

    flush   main: gather on GPU, async D2H into a pinned buffer, record event
            worker: wait for the event, scatter the pinned buffer into the CPU store
    fetch   main: select on GPU, async D2H of the block ids, record event
            worker: wait, diff against the resident set, gather the missing blocks from
                    the CPU store into a pinned buffer, async H2D into a GPU staging
                    buffer, record an event the main thread waits on at the swap

ONE worker thread, FIFO: a flush submitted for window k is scattered into the CPU store
before any later fetch job runs, so a fetch can never read a token the flush has not
written yet. (It could not anyway -- a token only becomes fetchable once it falls out of
the ~1000-token recency floor, four or more windows later -- but the ordering makes that
independent of the floor size.)

Nothing here is used unless the run asks for it: with VLLM_CASCADE_G=0 and
VLLM_CASCADE_ASYNC_FLUSH=0 (both defaults) the backend takes exactly the paths it took
before this file existed.

Tested on a GPU in isolation by fork_tests/test_async_io_gpu.py (inside inference_mode, as in
the engine); the scheduling arithmetic on a CPU by fork_tests/test_gap.py.
"""

import queue
import threading

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class PinnedPool:
    """Page-locked host buffers, recycled rather than reallocated.

    A buffer is handed out to one job at a time and returned when that job is done, so a
    DMA still reading it can never be overwritten: the worker returns it only after the
    copy that reads it has completed.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self._free: dict[int, list[torch.Tensor]] = {}
        self._lock = threading.Lock()
        self.allocated_rows = 0
        self.allocated_bytes = 0        # buffers are never freed, so this is the pool's footprint

    def acquire(self, rows: int, cols: int) -> torch.Tensor:
        rows = max(rows, 1)
        with self._lock:
            bucket = self._free.get(cols, [])
            for i, buf in enumerate(bucket):
                if buf.shape[0] >= rows:
                    bucket.pop(i)
                    return buf[:rows]
            self.allocated_rows += rows
            self.allocated_bytes += rows * cols * torch.empty((), dtype=self.dtype).element_size()
        return torch.empty(rows, cols, dtype=self.dtype, pin_memory=True)[:rows]

    def release(self, view: torch.Tensor) -> None:
        """Give back the buffer a view was cut from (`._base` when it is a slice)."""
        buf = view._base if view._base is not None else view
        with self._lock:
            self._free.setdefault(buf.shape[1], []).append(buf)


class DevicePool:
    """GPU staging for fetched blocks, held from the fetch until the swap consumes it.

    A buffer goes back into the pool with the event recorded after the scatter that read
    it, and is only handed out again once that event has completed -- the same rule as
    the pinned pool, on the device side.
    """

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype, self.device = dtype, device
        self._free: dict[int, list[tuple[torch.Tensor, torch.cuda.Event | None]]] = {}
        self._lock = threading.Lock()
        self.allocated_rows = 0
        self.allocated_bytes = 0        # GPU memory held by the pool; NOT part of vLLM's KV-cache budget

    def acquire(self, rows: int, cols: int) -> torch.Tensor:
        rows = max(rows, 1)
        with self._lock:
            bucket = self._free.get(cols, [])
            for i, (buf, event) in enumerate(bucket):
                if buf.shape[0] >= rows and (event is None or event.query()):
                    bucket.pop(i)
                    return buf[:rows]
            self.allocated_rows += rows
            self.allocated_bytes += rows * cols * torch.empty((), dtype=self.dtype).element_size()
        return torch.empty(rows, cols, dtype=self.dtype, device=self.device)[:rows]

    def release(self, view: torch.Tensor, event: torch.cuda.Event | None) -> None:
        buf = view._base if view._base is not None else view
        with self._lock:
            self._free.setdefault(buf.shape[1], []).append((buf, event))


class PendingFetch:
    """One (request, layer) refresh whose data is on its way to the GPU.

    Created at the selection step; consumed G decode steps later by the swap. `issued` is
    set once the worker has recorded `event`, so the main thread knows the copy exists
    before it makes its own stream wait on that event.
    """

    __slots__ = ("n_apply", "sel_new", "fs", "lf", "counts", "issued", "event", "staged",
                 "dst_slots", "dst_groups", "rows_fetched", "failed", "resident")

    def __init__(self, n_apply: int, sel_new: torch.Tensor, fs: int, lf: int,
                 counts: torch.Tensor) -> None:
        self.n_apply = n_apply          # seq_len the swap must happen at
        self.sel_new = sel_new          # [Hkv] int32 selected tokens per group after the swap
        self.fs = fs                    # floor start at apply time (selection used it too)
        self.lf = lf                    # floor length at apply time
        self.counts = counts            # [Hkv] long, blocks kept per group
        self.issued = threading.Event()
        self.event: torch.cuda.Event | None = None
        self.staged: torch.Tensor | None = None      # [rows, 2D] GPU, the fetched blocks
        self.dst_slots: torch.Tensor | None = None   # [rows] slot within the group's frame
        self.dst_groups: torch.Tensor | None = None  # [rows] KV group of each row
        self.rows_fetched = 0
        self.failed: BaseException | None = None
        # The slot bookkeeping AS IT WILL BE once this selection is swapped in. The worker
        # updates this copy, never the request's live one: a selection that is discarded
        # (the request never reached its swap step) must leave no trace, or the next
        # refresh would believe blocks are resident that were never fetched.
        self.resident: torch.Tensor | None = None

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the worker has issued the copy. Returns False on timeout."""
        return self.issued.wait(timeout)


class AsyncIO:
    """The worker thread and the two side streams. One instance per runtime."""

    def __init__(self, dtype: torch.dtype, device: torch.device) -> None:
        self.dtype, self.device = dtype, device
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.host_pool = PinnedPool(dtype)
        self.id_pool = PinnedPool(torch.int64)      # selected block ids, not KV data
        self.dev_pool = DevicePool(dtype, device)
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="cascade-io", daemon=True)
        self._thread.start()
        self.jobs_done = 0
        # Jobs that raised. A failed job is logged, not propagated -- the decode step must not
        # die because a background copy did -- so without this counter a broken worker is
        # invisible: the very first GPU run had every flush fail while the tokens still
        # matched (nothing re-reads a flushed token within 64 steps). Anything that judges a
        # run must check this is 0.
        self.failed_jobs = 0

    # ---- worker --------------------------------------------------------------
    def _run(self) -> None:
        # inference_mode is THREAD-LOCAL. vLLM runs the forward pass under it, so the CPU
        # store, the slot bookkeeping and the pinned buffers the main thread hands over are
        # "inference tensors", and a thread outside inference mode may not write into them
        # ("Inplace update to inference tensor outside InferenceMode"). Every job runs
        # inside it for that reason.
        with torch.inference_mode():
            while True:
                job = self._queue.get()
                if job is None:
                    return
                try:
                    job()
                except BaseException:            # never let the worker die silently
                    self.failed_jobs += 1
                    logger.exception("cascade: async I/O job failed")
                finally:
                    self.jobs_done += 1
                    self._queue.task_done()
                    if self.jobs_done % 512 == 0:
                        # The device pool is GPU memory vLLM does not know about: its KV cache was
                        # sized to a fraction of the card BEFORE this pool existed, so whatever it
                        # grows to has to fit in the leftover headroom. It only ever grows.
                        logger.info("cascade: async pools after %d jobs: device %.0f MiB (unbudgeted GPU), "
                                    "pinned host %.0f MiB", self.jobs_done, self.dev_pool.allocated_bytes / 2**20,
                                    (self.host_pool.allocated_bytes + self.id_pool.allocated_bytes) / 2**20)

    def drain(self) -> None:
        """Wait until every submitted job has run (used on fallback batches and teardown)."""
        self._queue.join()

    # ---- flush: GPU -> pinned host -> CPU store ------------------------------
    def submit_flush(self, packed: torch.Tensor, targets: list) -> None:
        """packed: [rows, 2D] on the GPU, already gathered in CPU-store order.
        targets: (cpu_store_tensor, group_count, start_token, token_count) per contiguous
        piece, in the same row order as `packed`.

        The D2H copy runs on its own stream (after waiting for the gather that produced
        `packed`), and the host-side scatter runs on the worker thread.
        """
        host = self.host_pool.acquire(packed.shape[0], packed.shape[1])
        event = torch.cuda.Event()
        self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.d2h_stream):
            host.copy_(packed, non_blocking=True)
            # `packed` was produced on the main stream; keep the allocator from reusing it
            # while this copy is still reading it.
            packed.record_stream(self.d2h_stream)
            event.record(self.d2h_stream)

        def job() -> None:
            event.synchronize()
            at = 0
            for store, groups, start, count in targets:
                store[:, start:start + count].copy_(host[at:at + groups * count].view(groups, count, -1))
                at += groups * count
            self.host_pool.release(host)

        self._queue.put(job)

    # ---- fetch: select on GPU -> diff on host -> pinned host -> GPU staging ---
    def submit_fetch(self, pending: PendingFetch, ids_gpu: torch.Tensor, store: torch.Tensor,
                     resident: torch.Tensor, row_elems: int, sel_block: int) -> None:
        """ids_gpu: [Hkv, K] selected block ids. store: this (request, layer)'s CPU store,
        [Hkv, tokens, 2D]. resident: [Hkv, K] CPU int64, block id in each selected slot --
        updated in place by the diff, which is what makes the next refresh incremental.

        The ids come back through a pinned buffer on the D2H stream, so the main thread
        never syncs on them; everything after that happens on the worker.
        """
        from vllm.cascade.ops.layout import refresh_slots

        groups, k_max = ids_gpu.shape
        ids_host = self.id_pool.acquire(groups, max(k_max, 1))
        ids_event = torch.cuda.Event()
        self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.d2h_stream):
            if k_max:
                ids_host[:, :k_max].copy_(ids_gpu, non_blocking=True)
                ids_gpu.record_stream(self.d2h_stream)
            ids_event.record(self.d2h_stream)

        def job() -> None:
            try:
                ids_event.synchronize()
                ids = ids_host[:, :k_max]
                pending.resident = resident
                need_rows, slot_rows, group_rows = [], [], []
                blocks = store.view(groups, -1, row_elems)
                for g in range(groups):
                    need, free = refresh_slots(resident[g], ids[g], int(pending.counts[g]))
                    if not need.numel():
                        continue
                    need_rows.append((g, need))
                    slot_rows.append((free[:, None] * sel_block
                                      + torch.arange(sel_block)[None, :]).reshape(-1))
                    group_rows.append(torch.full((free.numel() * sel_block,), g, dtype=torch.int64))
                total = sum(n.numel() for _, n in need_rows)
                if total:
                    host = self.host_pool.acquire(total, row_elems)
                    at = 0
                    for g, need in need_rows:
                        torch.index_select(blocks[g], 0, need, out=host[at:at + need.numel()])
                        at += need.numel()
                    staged = self.dev_pool.acquire(total * sel_block, row_elems // sel_block)
                    event = torch.cuda.Event()
                    with torch.cuda.stream(self.h2d_stream):
                        staged.view(total, row_elems).copy_(host, non_blocking=True)
                        event.record(self.h2d_stream)
                    pending.staged = staged
                    pending.event = event
                    pending.dst_slots = torch.cat(slot_rows)
                    pending.dst_groups = torch.cat(group_rows)
                    pending.rows_fetched = total * sel_block
                    # The pinned buffer is only free once the copy reading it has finished.
                    event.synchronize()
                    self.host_pool.release(host)
            except BaseException as e:       # the swap falls back to holding the working set
                pending.failed = e
                self.failed_jobs += 1
                logger.exception("cascade: async fetch failed; the swap will hold the working set")
            finally:
                self.id_pool.release(ids_host)
                pending.issued.set()

        self._queue.put(job)
