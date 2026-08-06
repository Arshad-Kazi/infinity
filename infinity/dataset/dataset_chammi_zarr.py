"""
CHAMMI single-cell Zarr dataset for Infinity.

Store layout (CHAMMI.zarr): one group per source dataset, each holding a 5-D uint8 array

    (images, cells, channels, height, width)

plus a 1-D `image_ids` string array indexed by the first axis. The `cells` axis is 1 --
each entry is already a single cell crop. Shapes differ per source:

    Allen  31,060 x 1 x 3 @ 238x374     ->  93,180 single-channel samples
    CP     36,360 x 1 x 5 @ 160x160     -> 181,800
    HPA    32,725 x 1 x 4 @ 512x512     -> 130,900
                                           -------
                                           405,880

One sample is one *channel* of one cell, matching how MorphEm consumes data (it is a
single-channel encoder). The channel is replicated to RGB for Infinity's VAE, which is
fixed at in_channels=3.

Differences from Infinity's T2I loader this replaces:
  * crops are square, so h_div_w is always 1.0 -- no aspect-ratio bucketing
  * there is no text; the second tuple element carries provenance strings, which the
    MorphEm conditioning path ignores

Contract required by train.py (DataLoader is built with batch_size=None, so the dataset
yields whole batches): __iter__ -> (images_B3HW, captions), __len__, set_epoch,
total_samples, and an h_div_w_template2generator mapping whose keys become
args.train_h_div_w_list.
"""

import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

CHAMMI_DATASETS = ("Allen", "CP", "HPA")

# every crop is squared, so a single aspect-ratio bucket covers the whole dataset
H_DIV_W_TEMPLATE = '1.000'


class ChammiZarrIterableDataset(IterableDataset):
    """Streams batches of single-channel CHAMMI crops, replicated to RGB.

    Args:
        data_path: path to CHAMMI.zarr
        pn: Infinity resolution key ('0.06M', '0.25M', ...). When image_size is None the
            output side length is taken from dynamic_resolution_h_w[1.0][pn]['pixel'],
            which is what keeps the emitted images consistent with the scale schedule
            the model builds for this pn.
        image_size: explicit override for the output side length.
        batch_size: per-GPU batch size; __iter__ yields tensors of exactly this size.
        num_replicas / rank: DDP world size and rank.
        dataloader_workers: must match DataLoader(num_workers=...), since the flat index
            is split across (rank, worker) pairs.
        datasets: subset of {"Allen", "CP", "HPA"}; defaults to all present.
        channels: restrict to these channel indices; indices beyond a dataset's channel
            count are skipped for that dataset.
        random_flip: random horizontal flip. Safe to leave on -- the MorphEm conditioning
            token is computed from the same flipped tensor inside the training loop, so
            there is no augmentation/cached-feature mismatch.
        read_threads: concurrent zarr reads per batch. Random reads on network storage are
            latency-bound (~78 ms each), so fetching a batch's samples in parallel is worth
            far more than it costs.
    """

    def __init__(
        self,
        data_path: str,
        pn: str = '0.06M',
        image_size: Optional[int] = None,
        batch_size: int = 1,
        num_replicas: int = 1,
        rank: int = 0,
        dataloader_workers: int = 1,
        datasets: Optional[Sequence[str]] = None,
        channels: Optional[Sequence[int]] = None,
        random_flip: bool = True,
        read_threads: int = 8,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.data_path = osp.abspath(osp.expanduser(data_path))
        self.pn = pn
        if image_size is None:
            image_size = dynamic_resolution_h_w[1.0][pn]['pixel'][0]
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.dataloader_workers = max(1, int(dataloader_workers))
        self.random_flip = random_flip
        self.read_threads = max(1, int(read_threads))
        self.seed = int(seed)
        self.epoch = 0

        self.worker_id = 0
        self.global_worker_id = self.rank
        self.global_workers = self.num_replicas * self.dataloader_workers
        self._handles = None  # zarr handles are opened lazily, per worker

        import zarr
        root = zarr.open(self.data_path, mode='r')
        requested = list(datasets) if datasets is not None else list(CHAMMI_DATASETS)
        names = [n for n in requested if n in root]
        if not names:
            raise ValueError(f'None of {requested} found in {self.data_path} (has: {list(root)})')

        # Build the flat (dataset, image, cell, channel) index from array metadata only --
        # no pixel data is touched here.
        self.dataset_names = names
        self.dataset_shapes = {}
        blocks = []
        for di, name in enumerate(names):
            shape = root[name][name].shape
            if len(shape) != 5:
                raise ValueError(f'Expected {name} to be (images, cells, channels, h, w), got {shape}')
            n_img, n_cell, n_chan = shape[:3]
            self.dataset_shapes[name] = shape
            keep = list(range(n_chan)) if channels is None else [c for c in channels if c < n_chan]
            if not keep:
                continue
            grid = np.stack(
                np.meshgrid(np.arange(n_img), np.arange(n_cell), keep, indexing='ij'), axis=-1
            ).reshape(-1, 3)
            block = np.empty((len(grid), 4), dtype=np.int32)
            block[:, 0] = di
            block[:, 1:] = grid
            blocks.append(block)
        if not blocks:
            raise ValueError(f'Channel filter {channels} selected no samples')
        self.index = np.concatenate(blocks, axis=0)

        # Batches per (rank, worker). Every worker must yield exactly this many, or the
        # DataLoader's round-robin over workers stalls and DDP ranks desynchronise.
        self.batches_per_worker = max(1, len(self.index) // self.global_workers // self.batch_size)

        # Single square bucket. train.py reads these keys into args.train_h_div_w_list.
        self.h_div_w_template2generator = {
            H_DIV_W_TEMPLATE: {
                'num_of_samples': len(self.index),
                'num_of_batches': self.batches_per_worker,
            }
        }
        print(f'[chammi] {self.data_path}: {len(self.index):,} single-channel samples, '
              f'image_size={self.image_size} (pn={pn}), batch_size={self.batch_size}, '
              f'world={self.num_replicas}x{self.dataloader_workers} workers, '
              f'batches/worker={self.batches_per_worker}')

    # ---------------------------------------------------------------- plumbing

    def __len__(self):
        # DataLoader(batch_size=None) round-robins workers, so an epoch is this many batches.
        return self.batches_per_worker * self.dataloader_workers

    def total_samples(self):
        return len(self) * self.num_replicas * self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getstate__(self):
        # zarr handles do not survive being pickled into worker processes
        state = self.__dict__.copy()
        state['_handles'] = None
        return state

    def _arrays(self):
        if self._handles is None:
            import zarr
            root = zarr.open(self.data_path, mode='r')
            self._handles = [(root[n][n], root[n]['image_ids']) for n in self.dataset_names]
        return self._handles

    def _set_worker_id(self):
        info = torch.utils.data.get_worker_info()
        self.worker_id = info.id if info else 0
        self.global_worker_id = self.rank * self.dataloader_workers + self.worker_id

    # ------------------------------------------------------------- reading

    def _preprocess(self, raw_hw: np.ndarray) -> torch.Tensor:
        """(H, W) uint8 -> (3, image_size, image_size) float32 in [-1, 1].

        Center-crops to a square first so non-square sources (Allen, 238x374) keep their
        aspect ratio instead of being squashed. The single channel is replicated to RGB
        because Infinity's VAE is fixed at in_channels=3.
        """
        x = torch.from_numpy(np.ascontiguousarray(raw_hw)).float()[None]  # (1, H, W)
        h, w = x.shape[-2:]
        if h != w:
            x = v2.functional.center_crop(x, min(h, w))
        if x.shape[-1] != self.image_size:
            x = v2.functional.resize(x, [self.image_size, self.image_size], antialias=True)
        if self.random_flip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[-1])
        x = x.div_(127.5).sub_(1.0)
        return x.expand(3, -1, -1).contiguous()

    def _load_one(self, row) -> Tuple[torch.Tensor, str]:
        di, img, cell, chan = (int(v) for v in row)
        arr, ids = self._arrays()[di]
        raw = np.asarray(arr[img, cell, chan])  # exactly one chunk: (H, W) uint8
        caption = f'{self.dataset_names[di]}|ch{chan}|{ids[img]}'
        return self._preprocess(raw), caption

    # ------------------------------------------------------------- iteration

    def _epoch_order(self) -> np.ndarray:
        """Rows assigned to this (rank, worker) for the current epoch.

        A single permutation is drawn from a seed shared by all workers, then strided, so
        each worker gets a disjoint slice and the assignment reshuffles every epoch.
        """
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(len(self.index))
        return perm[self.global_worker_id::self.global_workers]

    def __iter__(self):
        self._set_worker_id()
        order = self._epoch_order()
        need = self.batches_per_worker * self.batch_size
        if len(order) < need:  # keep every worker's batch count identical
            reps = int(np.ceil(need / max(1, len(order))))
            order = np.tile(order, reps)
        order = order[:need]

        pool = ThreadPoolExecutor(max_workers=min(self.read_threads, self.batch_size))
        try:
            for b in range(self.batches_per_worker):
                rows = self.index[order[b * self.batch_size:(b + 1) * self.batch_size]]
                results = list(pool.map(self._load_one, rows))
                images = torch.stack([r[0] for r in results])          # (B, 3, S, S)
                captions = [r[1] for r in results]
                yield images, captions
        finally:
            pool.shutdown(wait=False)

    # ------------------------------------------------------------- debugging

    def describe(self) -> str:
        lines = [f'{self.data_path}  ({len(self.index):,} single-channel samples)']
        for di, name in enumerate(self.dataset_names):
            n = int((self.index[:, 0] == di).sum())
            s = self.dataset_shapes[name]
            lines.append(f'  {name:6s} {n:>9,} samples   {s[0]:,} images x {s[1]} cells '
                         f'x {s[2]} channels @ {s[3]}x{s[4]}')
        lines.append(f'  -> emits (B, 3, {self.image_size}, {self.image_size}) in [-1, 1], '
                     f'h_div_w=1.0, {len(self)} batches/epoch/rank')
        return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    import time

    p = argparse.ArgumentParser(description='Inspect the CHAMMI Zarr dataset for Infinity.')
    p.add_argument('--data-path', type=str,
                   default='/hdd/jcaicedo/projects/dinov3/zarr_datasets/CHAMMI.zarr')
    p.add_argument('--pn', type=str, default='0.06M')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--datasets', type=str, nargs='*', default=None)
    p.add_argument('--channels', type=int, nargs='*', default=None)
    args = p.parse_args()

    ds = ChammiZarrIterableDataset(
        args.data_path, pn=args.pn, batch_size=args.batch_size,
        dataloader_workers=max(1, args.workers), datasets=args.datasets, channels=args.channels,
    )
    print(ds.describe())

    from torch.utils.data import DataLoader
    ld = DataLoader(ds, batch_size=None, num_workers=args.workers)
    t0 = time.time()
    n = 0
    for i, (imgs, caps) in enumerate(ld):
        if i == 0:
            print(f'\nfirst batch: {tuple(imgs.shape)} {imgs.dtype} '
                  f'range [{imgs.min():.2f}, {imgs.max():.2f}]')
            print(f'  h_div_w = {imgs.shape[-2] / imgs.shape[-1]:.3f}')
            for c in caps[:3]:
                print(f'  {c}')
        n += imgs.shape[0]
        if i + 1 >= args.batches:
            break
    dt = time.time() - t0
    print(f'\n{n} images in {dt:.1f}s -> {n / dt:.1f} img/s '
          f'({args.workers} workers x {ds.read_threads} read threads)')
