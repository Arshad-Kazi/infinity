"""
MorphEm image-conditioning encoder for Infinity.

Replaces the T5 text encoder. MorphEm (CaicedoLab/MorphEm) is a ViT-Small trained
with the DINO "Bag of Channels" recipe on CHAMMI-75 microscopy images. It consumes
*single-channel* images (in_chans=1, embed_dim=384), so a C-channel microscopy image
is encoded one channel at a time and yields C CLS tokens.

Those C CLS tokens play the role T5's token features used to play: they are packed
into the same varlen "compact" layout Infinity already consumes, so nothing inside
the transformer changes.

    (kv_compact, lens, cu_seqlens_k, max_seqlen)
        kv_compact:   (sum(lens), 384)  all samples' channel-CLS tokens, concatenated
        lens:         [C_1, ..., C_B]   channels per sample
        cu_seqlens_k: (B+1,) int32      cumulative offsets for flash-attn varlen
        max_seqlen:   int               max(lens)
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditioning tuple consumed by Infinity.forward / autoregressive_infer_cfg.
CondTuple = Tuple[torch.Tensor, List[int], torch.Tensor, int]

MORPHEM_DEFAULT_PATH = 'CaicedoLab/MorphEm'
MORPHEM_EMBED_DIM = 384
MORPHEM_IMG_SIZE = 224


def build_cond_tuple(cond_BLC: torch.Tensor, lens: Optional[Sequence[int]] = None) -> CondTuple:
    """Pack per-sample conditioning tokens into Infinity's varlen compact layout.

    :param cond_BLC: (B, L, C) conditioning tokens, right-padded when lens is given
    :param lens: valid token count per sample; None means all L tokens are valid
    """
    B, L, C = cond_BLC.shape
    if lens is None:
        lens = [L] * B
    lens = [int(x) for x in lens]
    assert len(lens) == B, f'{len(lens)=} != {B=}'
    assert max(lens) <= L, f'{max(lens)=} > {L=}'

    kv_compact = torch.cat([cond_BLC[i, :l] for i, l in enumerate(lens)], dim=0)
    cu_seqlens_k = F.pad(
        torch.tensor(lens, dtype=torch.int32, device=cond_BLC.device).cumsum(0, dtype=torch.int32), (1, 0)
    )
    return kv_compact, lens, cu_seqlens_k, max(lens)


class MorphEmEncoder(nn.Module):
    """Frozen MorphEm ViT that turns an image into per-channel CLS tokens.

    The wrapped model is always in eval mode with requires_grad=False; it is a fixed
    feature extractor, exactly like the T5 encoder it replaces.
    """

    def __init__(
        self,
        model_path: str = MORPHEM_DEFAULT_PATH,
        img_size: int = MORPHEM_IMG_SIZE,
        normalize: str = 'morphem',
        saturation_noise: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = 'cuda',
    ):
        """
        :param model_path: HF repo id or local path
        :param img_size: conditioning images are resized to img_size x img_size.
            MorphEm interpolates its position embedding, so other sizes work, but
            224 matches pre-training.
        :param normalize: 'morphem' reproduces the official preprocessing exactly
            (SaturationNoiseInjector -> PerImageNormalize -> Resize(224, antialias)),
            'none' passes the input through untouched.
        :param saturation_noise: apply SaturationNoiseInjector. It is part of the official
            pipeline, but it is stochastic (saturated pixels are replaced by uniform noise),
            so disable it if you need bit-exact reproducible conditioning.

        NOTE: inputs are expected in the [0, 255] intensity range, matching the official
        demo (and CHAMMI's uint8 source data). SaturationNoiseInjector keys off `== 255`,
        so a [0, 1] input would silently skip it.
        """
        super().__init__()
        from transformers import AutoModel

        self.model_path = model_path
        self.img_size = img_size
        self.normalize = normalize
        self.saturation_noise = saturation_noise
        assert normalize in ('morphem', 'none'), f'unknown normalize={normalize}'

        # PerImageNormalize from the model card: instance norm, no affine, no running stats.
        # Matches torch's biased variance and eps-inside-sqrt exactly.
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1, affine=False, track_running_stats=False, eps=1e-7
        )

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
        self.model.eval()
        self.model.requires_grad_(False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(device)

        self.embed_dim: int = self.model.config.embed_dim
        self.in_chans: int = self.model.config.in_chans
        assert self.in_chans == 1, (
            f'MorphEm is expected to be single-channel (bag-of-channels), got in_chans={self.in_chans}'
        )
        print(f'[MorphEm] loaded {model_path}: embed_dim={self.embed_dim}, img_size={img_size}, normalize={normalize}')

    @property
    def cond_dim(self) -> int:
        return self.embed_dim

    def train(self, mode: bool = True):
        # A frozen feature extractor must never leave eval mode, whatever the training loop does.
        return super().train(False)

    @staticmethod
    def _inject_saturation_noise(x_N1HW: torch.Tensor, low: float = 200., high: float = 255.) -> torch.Tensor:
        """SaturationNoiseInjector: replace saturated (==255) pixels with U(low, high).

        Equivalent to the model card's zero-then-add-masked-noise formulation. Applied to
        every sample; the card's `x[0]` indexing is written for a single (1, H, W) image,
        which is how it is used inside their per-sample training transform.
        """
        noise = torch.empty_like(x_N1HW).uniform_(low, high)
        return torch.where(x_N1HW == 255, noise, x_N1HW)

    def preprocess(self, img_BCHW: torch.Tensor) -> torch.Tensor:
        """Official MorphEm preprocessing, applied per (sample, channel).

        Order matters and follows the model card exactly:
            SaturationNoiseInjector -> PerImageNormalize -> Resize(img_size, antialias=True)

        :param img_BCHW: (B, C, H, W) in the [0, 255] intensity range
        :return: (B*C, 1, S, S), the single-channel batch MorphEm consumes
        """
        from torchvision.transforms import v2

        assert img_BCHW.dim() == 4, f'expected (B, C, H, W), got {tuple(img_BCHW.shape)}'
        B, C, H, W = img_BCHW.shape
        # fold channels into the batch dim first, so normalization is exactly per-channel
        x = img_BCHW.float().reshape(B * C, 1, H, W)

        if self.normalize == 'none':
            return x
        if self.saturation_noise:
            x = self._inject_saturation_noise(x)
        x = self.instance_norm(x)
        if (H, W) != (self.img_size, self.img_size):
            x = v2.functional.resize(x, [self.img_size, self.img_size], antialias=True)
        return x

    @torch.no_grad()
    def encode(self, img_BCHW: torch.Tensor, already_preprocessed: bool = False) -> torch.Tensor:
        """Bag-of-channels encode: each channel goes through the ViT separately.

        :param img_BCHW: (B, C, H, W) conditioning image in [0, 255], C = #microscopy channels
        :return: (B, C, embed_dim) per-channel CLS tokens
        """
        assert img_BCHW.dim() == 4, f'expected (B, C, H, W), got {tuple(img_BCHW.shape)}'
        B, C = img_BCHW.shape[:2]
        flat = self.preprocess(img_BCHW) if not already_preprocessed else img_BCHW.reshape(
            B * C, 1, *img_BCHW.shape[-2:]
        )
        p = next(self.model.parameters())
        flat = flat.to(device=p.device, dtype=p.dtype)
        # forward_features is the reliable CLS API; model.forward(return_dict=True) discards the pooled output
        cls = self.model.forward_features(flat)['x_norm_clstoken']  # (B*C, embed_dim)
        return cls.reshape(B, C, self.embed_dim).float()

    @torch.no_grad()
    def forward(self, img_BCHW: torch.Tensor, already_preprocessed: bool = False) -> CondTuple:
        """Encode an image batch straight into Infinity's conditioning tuple."""
        cond_BLC = self.encode(img_BCHW, already_preprocessed=already_preprocessed)
        return build_cond_tuple(cond_BLC)


def gen_img_to_cond_img(inp_BCHW: torch.Tensor) -> torch.Tensor:
    """Map a generation-target image in [-1, 1] to the [0, 255] range MorphEm expects.

    The dataloader normalizes generation targets to [-1, 1] (dataset_t2i_iterable.transform,
    and CHAMMI's `x.div_(127.5).sub_(1.0)`); this is the exact inverse. The [0, 255] scale
    matters: SaturationNoiseInjector keys off `pixel == 255`, so feeding [0, 1] would
    silently disable it and put the input off-distribution.
    """
    return inp_BCHW.float().add(1).mul_(127.5).clamp_(0, 255)
