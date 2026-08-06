"""
Periodic sample logging to Weights & Biases during Infinity training.

Every `--sample_every` iterations a fixed set of reference cells is reconstructed from
its own MorphEm CLS tokens and logged as one image: reference cells on the top row,
reconstructions directly underneath. The reference batch never changes, so scrolling the
wandb panel over time shows generation quality evolving on the same cells.

Design constraints this file respects:
  * Nothing here runs inside the training graph. Sampling is under torch.no_grad(), the
    model's train/eval mode is saved and restored, and the RNG is left untouched.
  * Under FSDP the parameters are sharded, so sampling runs inside summon_full_params.
    That is a collective: *every* rank must enter it, even though only rank 0 logs.
  * Any failure is caught and reported. A visualization problem must never kill a
    training run that is otherwise healthy.
"""

import traceback
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from infinity.utils import wandb_utils
from infinity.models.morphem_encoder import gen_img_to_cond_img


def _to_uint8_grid(ref_B3HW: torch.Tensor, gen_B3HW: torch.Tensor, pad: int = 2) -> np.ndarray:
    """Stack references over reconstructions into a single (H, W, 3) uint8 array.

    Both inputs are (B, 3, H, W) in [-1, 1]. Columns are cells; row 0 is the reference and
    row 1 the reconstruction of that same cell.
    """
    def prep(t):
        t = t.detach().float().add(1).div(2).clamp(0, 1).cpu()
        return F.pad(t, (pad, pad, pad, pad), value=1.0)

    rows = [prep(ref_B3HW), prep(gen_B3HW)]
    grid = torch.cat([torch.cat(list(r), dim=-1) for r in rows], dim=-2)  # (3, 2H, B*W)
    return grid.mul(255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


class SampleLogger:
    """Holds the fixed reference batch and renders samples on a fixed interval."""

    def __init__(
        self,
        cond_img_B3HW: torch.Tensor,
        captions: List[str],
        scale_schedule,
        every: int = 0,
        cfg: float = 3.0,
        tau: float = 1.0,
        top_k: int = 900,
        top_p: float = 0.97,
        cfg_insertion_layer: int = -5,
        vae_type: int = 1,
        cond_channels: int = 1,
    ):
        self.cond_img = cond_img_B3HW
        self.captions = captions
        self.scale_schedule = scale_schedule
        self.every = int(every)
        self.cfg, self.tau = cfg, tau
        self.top_k, self.top_p = top_k, top_p
        self.cfg_insertion_layer = cfg_insertion_layer
        self.vae_type = vae_type
        self.cond_channels = cond_channels
        self.enabled = self.every > 0 and cond_img_B3HW is not None

    def should_log(self, g_it: int) -> bool:
        return self.enabled and (g_it + 1) % self.every == 0

    @torch.no_grad()
    def _generate(self, trainer, cond_encoder) -> Optional[torch.Tensor]:
        """Reconstruct the reference batch from its own CLS tokens. Returns (B, 3, H, W)."""
        gpt = trainer.gpt_wo_ddp
        n = self.cond_img.shape[0]

        src = self.cond_img if self.cond_channels <= 0 else self.cond_img[:, :self.cond_channels]
        cond_tuple = cond_encoder(gen_img_to_cond_img(src))

        nscale = len(self.scale_schedule)
        # ret_img=True returns uint8 (B, H, W, 3) with the channel order flipped to BGR
        _, _, img_BHW3 = gpt.autoregressive_infer_cfg(
            vae=trainer.vae_local,
            scale_schedule=self.scale_schedule,
            label_B_or_BLT=cond_tuple,
            B=n,
            cfg_list=[self.cfg] * nscale,
            tau_list=[self.tau] * nscale,
            top_k=self.top_k, top_p=self.top_p,
            cfg_insertion_layer=[self.cfg_insertion_layer],
            vae_type=self.vae_type, returns_vemb=1,
            ret_img=True, inference_mode=True, g_seed=0,
        )
        img = img_BHW3.flip(dims=(3,))                       # BGR -> RGB
        img = img.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)  # -> (B, 3, H, W) in [-1, 1]
        return img

    def log(self, trainer, cond_encoder, g_it: int, args=None):
        """Generate and log. Safe to call on every rank; only rank 0 writes to wandb."""
        if not self.enabled:
            return
        import infinity.utils.dist as dist

        gpt = trainer.gpt_wo_ddp
        was_training = gpt.training
        try:
            gpt.eval()
            use_fsdp = bool(getattr(args, 'zero', 0)) if args is not None else False
            if use_fsdp:
                # collective: all ranks must enter, or the ones that skip it will hang
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.summon_full_params(trainer.gpt, recurse=True, writeback=False):
                    gen = self._generate(trainer, cond_encoder)
            else:
                gen = self._generate(trainer, cond_encoder)

            if dist.is_master() and gen is not None:
                grid = _to_uint8_grid(self.cond_img[:, :3], gen)
                wandb_utils.wandb.log(
                    {'samples/reconstructions': wandb_utils.wandb.Image(
                        grid,
                        caption=f'iter {g_it + 1} | top: reference cells | '
                                f'bottom: reconstructed from MorphEm CLS (cfg={self.cfg})',
                    )},
                    step=g_it,
                )
                print(f'[sample_vis] logged {gen.shape[0]} reconstructions at g_it={g_it + 1}', flush=True)
        except Exception as e:
            # never let visualization take down a healthy training run
            print(f'[sample_vis] SKIPPED at g_it={g_it + 1}: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            if was_training:
                gpt.train()


def build_sample_logger(args, dataset, scale_schedule) -> SampleLogger:
    """Draw the fixed reference batch and construct the logger. Returns a disabled logger
    when sampling is off or the dataset cannot provide a fixed batch."""
    if getattr(args, 'sample_every', 0) <= 0 or not hasattr(dataset, 'fixed_batch'):
        return SampleLogger(None, [], scale_schedule, every=0)
    cond_img, captions = dataset.fixed_batch(args.sample_n)
    cond_img = cond_img.to(args.device)
    print(f'[sample_vis] fixed reference batch: {tuple(cond_img.shape)}, every {args.sample_every} iters')
    for c in captions:
        print(f'[sample_vis]   {c}')
    return SampleLogger(
        cond_img, captions, scale_schedule,
        every=args.sample_every, cfg=args.sample_cfg, tau=args.sample_tau,
        top_k=args.sample_top_k, top_p=args.sample_top_p,
        cfg_insertion_layer=args.cfg_insertion_layer,
        vae_type=1 if args.vae_type != 0 else 0,
        cond_channels=getattr(args, 'morphem_cond_channels', 1),
    )
