"""C9 paired translator network.

Encoders, UNI fusion, decoders, spectral-norm critics, PatchNCE, PairNCE, and the VIB Gaussian head.
Copied from bbdm_BCI/train_c9_paired.py. The training loop, dataset code, and Hugging Face upload helpers are not included.
The original module reads a token file on import. This file does not. Pass a UNI module into C9Paired; nothing here downloads weights.

VIB is C9Paired(vib=True) with GaussianBottleneckHead. vib/network.py imports those classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN


def haar_dwt_1level(x):
    """Single-level Haar DWT. x: [B, C, H, W] with H, W even.
    Returns dict {LL, LH, HL, HH} each [B, C, H/2, W/2]. Supports double-backward.
    """
    a = x[..., 0::2, 0::2]; b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]; d = x[..., 1::2, 1::2]
    return dict(
        LL=(a + b + c + d) * 0.5,
        LH=(a - b + c - d) * 0.5,
        HL=(a + b - c - d) * 0.5,
        HH=(a - b - c + d) * 0.5,
    )


def haar_hf_concat(x):
    """Return concat(LH, HL, HH) at level-1. x: [B, 3, H, W] -> [B, 9, H/2, W/2]."""
    b = haar_dwt_1level(x)
    return torch.cat([b["LH"], b["HL"], b["HH"]], dim=1)


# Spatial lattice from --z_hw. Same 3 stem convs; strides only.
# S128 (1,2,1) is the locked constructor. Do not add stem layers.
Z_HW_ALLOWED = (32, 64, 128, 256)
Z_HW_STEM_STRIDES = {
    256: (1, 1, 1),
    128: (1, 2, 1),
    64: (1, 2, 2),
    32: (2, 2, 2),
}


def derive_spatial(z_hw):
    """Return (z_hw, stem_strides, upsample_factor) for a native lattice.

    Input stays 256^2. Stem product of strides is 256/z_hw. Decoder nearest
    factor is 256/z_hw (S256 x1, S128 x2, S64 x4, S32 x8).
    """
    z_hw = int(z_hw)
    if z_hw not in Z_HW_STEM_STRIDES:
        raise ValueError(f"z_hw must be one of {Z_HW_ALLOWED}, got {z_hw}")
    strides = Z_HW_STEM_STRIDES[z_hw]
    return z_hw, strides, 256 // z_hw

class DilatedResBlock(nn.Module):
    """Pre-activation ResBlock: GN -> GeLU -> Conv3(dilated) -> GN -> GeLU -> Conv3(un-dilated).

    First conv uses dilation d (rate); second conv uses dilation 1.
    Sawtooth dilation pattern [1, 2, 5, 1, 2, 5, ...] across a stack of blocks
    avoids the gridding artifact (Wang et al. 2018, HDC).
    """
    def __init__(self, ch, d=1, groups=32):
        super().__init__()
        gn_groups = min(groups, ch)
        self.gn1 = nn.GroupNorm(gn_groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
        self.gn2 = nn.GroupNorm(gn_groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)

    def forward(self, x):
        h = self.conv1(F.gelu(self.gn1(x)))
        h = self.conv2(F.gelu(self.gn2(h)))
        return x + h


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2, CVPR 2023).

    Channel competition: amplifies channels whose global L2 norm exceeds the
    mean across channels, suppresses below-mean channels. Starts as identity
    (gamma=beta=0). Learned during training.
    """
    def __init__(self, ch):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)            # [B, C, 1, 1]
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)              # relative-to-mean
        return self.gamma * (x * Nx) + self.beta + x


class DilatedResBlockGRN(nn.Module):
    """DilatedResBlock + GRN (after conv1) + LayerScale on residual branch.

    Fixes measured rank collapse: trained conv1 learned to respond to only
    ~14 of 128 input directions, and h dominates x with ||h||/||x|| ≈ 4.
    GRN encourages channel competition (breaks learned redundancy).
    LayerScale (learnable per-channel gamma, init 0.1) prevents low-rank h
    from overwriting the high-rank identity.
    """
    def __init__(self, ch, d=1, groups=32, layer_scale_init=0.1, use_layer_scale=True):
        super().__init__()
        gn_groups = min(groups, ch)
        self.gn1   = nn.GroupNorm(gn_groups, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=d, dilation=d)
        self.grn   = GRN(ch)
        self.gn2   = nn.GroupNorm(gn_groups, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init * torch.ones(1, ch, 1, 1))
        else:
            # Identity: not learnable. Rank tracking still sees gamma=1.
            self.register_buffer("gamma", torch.ones(1, ch, 1, 1))

    def forward(self, x):
        h = self.conv1(F.gelu(self.gn1(x)))
        h = self.grn(h)
        h = self.conv2(F.gelu(self.gn2(h)))
        if self.use_layer_scale:
            return x + self.gamma * h
        return x + h


class C9Encoder(nn.Module):
    """C9 encoder: stem + 4 dilated ResBlocks @ computational_ch, optional 1x1 bottle.

    Sawtooth dilation [1, 2, 5, 1]. Locked: computational_ch=z_ch=512 (no bottle).
    Failed z64 (mix at 64): computational_ch=512, z_ch=64, expand_before_fuse=False
      -> output 64-ch into UNI fuse (concat 320).
    Cleaner z64: same bottle, expand_before_fuse=True
      -> 512 -> 64 -> 512, then locked UNI fuse (concat 768).
    Default body is plain DilatedResBlock. `--use_grn_encoder` swaps the 4 body
    blocks to DilatedResBlockGRN. Stem stays GroupNorm (not GRN).
    """
    def __init__(self, in_ch=3, base_ch=128, mid_ch=256,
                 computational_ch=512, z_ch=None, out_ch=None,
                 expand_before_fuse=False, dilations=None,
                 use_grn=False, layer_scale_init=0.1, use_layer_scale=True,
                 stem_strides=(1, 2, 1)):
        super().__init__()
        # out_ch kept as alias for computational_ch (older call sites).
        if out_ch is not None:
            computational_ch = out_ch
        if z_ch is None:
            z_ch = computational_ch
        self.z_ch = int(z_ch)
        self.computational_ch = int(computational_ch)
        self.expand_before_fuse = bool(expand_before_fuse)
        self.use_grn = bool(use_grn)
        self.dilations = tuple(int(d) for d in (dilations or (1, 2, 5, 1)))
        if len(self.dilations) != 4:
            raise ValueError(f"encoder dilations must be length 4, got {self.dilations}")
        self.stem_strides = tuple(int(s) for s in stem_strides)
        if len(self.stem_strides) != 3:
            raise ValueError(f"stem_strides must be length 3, got {self.stem_strides}")
        self.out_ch = (self.computational_ch
                       if self.expand_before_fuse and self.z_ch != self.computational_ch
                       else self.z_ch)
        s1, s2, s3 = self.stem_strides
        # S128 lock is (1, 2, 1): conv1 default stride 1, conv2 stride 2, conv3 stride 1.
        self.stem_conv1 = nn.Conv2d(in_ch, base_ch, 7, stride=s1, padding=3)
        self.stem_gn1 = nn.GroupNorm(min(32, base_ch), base_ch)
        self.stem_conv2 = nn.Conv2d(base_ch, mid_ch, 3, stride=s2, padding=1)
        self.stem_gn2 = nn.GroupNorm(min(32, mid_ch), mid_ch)
        self.stem_conv3 = nn.Conv2d(mid_ch, computational_ch, 3, stride=s3, padding=1)
        self.stem_gn3 = nn.GroupNorm(min(32, computational_ch), computational_ch)
        # Stem stays GroupNorm (not a ResBlock). GRN+LS only on the 4 body blocks.
        Blk = (lambda ch, d: DilatedResBlockGRN(
            ch, d, layer_scale_init=layer_scale_init, use_layer_scale=use_layer_scale)) \
            if self.use_grn else DilatedResBlock
        self.blocks = nn.ModuleList([
            Blk(computational_ch, d=d) for d in self.dilations
        ])
        # True bottleneck only when z_ch differs (preserves locked 512 ckpt layout).
        self.bottle = (nn.Conv2d(computational_ch, self.z_ch, 1)
                       if self.z_ch != computational_ch else None)
        self.expand = (nn.Conv2d(self.z_ch, self.computational_ch, 1)
                       if self.bottle is not None and self.expand_before_fuse else None)

    def forward(self, x):
        x = F.gelu(self.stem_gn1(self.stem_conv1(x)))
        x = F.gelu(self.stem_gn2(self.stem_conv2(x)))
        x = F.gelu(self.stem_gn3(self.stem_conv3(x)))
        for blk in self.blocks:
            x = blk(x)
        if self.bottle is not None:
            x = self.bottle(x)                            # z_ch x 128x128
        if self.expand is not None:
            x = self.expand(x)                            # computational_ch x 128x128
        return x


class UNIFeatureExtractor(nn.Module):
    """Frozen UNI-2H wrapper. Returns feature maps at requested taps.

    Same as C8. No gradient into UNI.
    """
    def __init__(self, uni_module, taps=(13, 19), target_size=224,
                  n_cls=1, n_reg=8):
        super().__init__()
        self.uni = uni_module
        self.taps = tuple(taps)
        self.target_size = target_size
        self.n_cls_reg = int(n_cls) + int(n_reg)
        self._cache = {}
        self._hooks = []
        for L in self.taps:
            self._hooks.append(uni_module.blocks[L].register_forward_hook(self._make_hook(L)))
        self.register_buffer("mean_uni", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std_uni", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _make_hook(self, L):
        def h(m, i, o): self._cache[L] = o
        return h

    def preprocess(self, x):
        x01 = (x + 1) * 0.5
        x224 = F.interpolate(x01, size=self.target_size, mode="bilinear", align_corners=False)
        return (x224 - self.mean_uni) / self.std_uni

    def forward(self, x):
        """x in [-1, 1] [B, 3, H, W]. Returns dict {L: [B, 1536, 16, 16]}."""
        B = x.size(0)
        x_pre = self.preprocess(x)
        _ = self.uni.forward_features(x_pre)
        out = {}
        for L in self.taps:
            tokens = self._cache[L][:, self.n_cls_reg:, :]     # [B, 256, 1536]
            hw = int(tokens.size(1) ** 0.5)
            out[L] = tokens.reshape(B, hw, hw, -1).permute(0, 3, 1, 2)
        return out


class UNIBottleneckFuse(nn.Module):
    """Fuse encoder z_enc (z_ch) with UNI L13+L19 into fused (out_ch).

    * 1x1 conv projects each UNI tap (1536ch) -> uni_proj_ch (128ch)
    * bilinear upsample UNI 16x16 -> target_hw (Z lattice; locked 128)
    * concat with z_enc -> z_ch + 2*uni_proj_ch
    * 1x1 conv to out_ch (locked: 512→512; z64 cell: 320→512)
    """
    def __init__(self, z_ch=512, out_ch=None, uni_ch=1536, uni_proj_ch=128,
                 taps=(13, 19), target_hw=128):
        super().__init__()
        if out_ch is None:
            out_ch = z_ch
        self.taps = tuple(taps)
        self.target_hw = target_hw
        self.z_ch = int(z_ch)
        self.out_ch = int(out_ch)
        self.proj_L13 = nn.Conv2d(uni_ch, uni_proj_ch, 1)
        self.proj_L19 = nn.Conv2d(uni_ch, uni_proj_ch, 1)
        cat_ch = z_ch + 2 * uni_proj_ch
        self.fuse = nn.Conv2d(cat_ch, out_ch, 1)
        self.gn = nn.GroupNorm(min(32, out_ch), out_ch)

    def forward(self, z_enc, uni_feats):
        """z_enc: [B, z_ch, H, H] with H=target_hw. Returns [B, out_ch, H, H]."""
        u13 = self.proj_L13(uni_feats[self.taps[0]])       # [B, 128, 16, 16]
        u19 = self.proj_L19(uni_feats[self.taps[1]])
        u13 = F.interpolate(u13, size=self.target_hw, mode="bilinear", align_corners=False)
        u19 = F.interpolate(u19, size=self.target_hw, mode="bilinear", align_corners=False)
        cat = torch.cat([z_enc, u13, u19], dim=1)
        return F.gelu(self.gn(self.fuse(cat)))


class C9Decoder(nn.Module):
    """C9 decoder: 5 dilated ResBlocks + upsample head (128->256).

    Sawtooth dilation [2, 5, 1, 2, 5].
    If `use_grn=True`, uses DilatedResBlockGRN (GRN; LayerScale unless off).
    """
    def __init__(self, in_ch=512, mid_ch=128, out_ch=3, use_grn=False,
                 layer_scale_init=0.1, use_layer_scale=True, dilations=None,
                 upsample_factor=2):
        super().__init__()
        self.dilations = tuple(int(d) for d in (dilations or (2, 5, 1, 2, 5)))
        if len(self.dilations) != 5:
            raise ValueError(f"decoder dilations must be length 5, got {self.dilations}")
        self.upsample_factor = int(upsample_factor)
        if self.upsample_factor < 1:
            raise ValueError(f"upsample_factor must be >= 1, got {self.upsample_factor}")
        Blk = (lambda ch, d: DilatedResBlockGRN(
            ch, d, layer_scale_init=layer_scale_init, use_layer_scale=use_layer_scale)) \
            if use_grn else DilatedResBlock
        self.blocks = nn.ModuleList([
            Blk(in_ch, d=d) for d in self.dilations
        ])
        # Upsample head Z -> 256 via nearest + Conv3. S128 lock is factor 2.
        self.up_conv = nn.Conv2d(in_ch, mid_ch, 3, padding=1)
        self.up_gn = nn.GroupNorm(min(32, mid_ch), mid_ch)
        # Final conv 7x7 -> 3ch
        self.head_conv = nn.Conv2d(mid_ch, out_ch, 7, padding=3)
        nn.init.xavier_normal_(self.head_conv.weight, gain=0.1)
        nn.init.zeros_(self.head_conv.bias)

    def forward(self, z):
        for blk in self.blocks:
            z = blk(z)
        if self.upsample_factor == 1:
            z_up = z
        else:
            z_up = F.interpolate(z, scale_factor=self.upsample_factor, mode="nearest")
        z_up = F.gelu(self.up_gn(self.up_conv(z_up)))       # [B, 128, 256, 256]
        return torch.tanh(self.head_conv(z_up))              # [B, 3, 256, 256]


class SNPatchGAN(nn.Module):
    """PatchGAN discriminator with spectral norm on all convs. No InstanceNorm.

    Input: concat([source, target_or_gen]) = 6ch.
    Output: [B, 1, 30, 30] patch logits.
    """
    def __init__(self, in_ch=6, base_ch=64):
        super().__init__()
        self.conv1 = SN(nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1))
        self.conv2 = SN(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1))
        self.conv3 = SN(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1))
        self.conv4 = SN(nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=1, padding=1))
        self.conv5 = SN(nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1))

    def forward(self, source, target_or_gen):
        x = torch.cat([source, target_or_gen], dim=1)       # [B, 6, 256, 256]
        x = F.leaky_relu(self.conv1(x), 0.2)                 # [B, 64, 128, 128]
        x = F.leaky_relu(self.conv2(x), 0.2)                 # [B, 128, 64, 64]
        x = F.leaky_relu(self.conv3(x), 0.2)                 # [B, 256, 32, 32]
        x = F.leaky_relu(self.conv4(x), 0.2)                 # [B, 512, 31, 31]
        return self.conv5(x)                                  # [B, 1, 30, 30]


class SNPatchGAN_HF(nn.Module):
    """PatchGAN on wavelet HF bands (Haar 1-level LH+HL+HH concat).

    Input: concat([source_HF, gen_or_target_HF]) = 18ch at 128x128 (half of pixel).
    Output: [B, 1, 15, 15] patch logits (one less downsample than pixel D since starting at 128x128).
    """
    def __init__(self, in_ch=18, base_ch=64):
        super().__init__()
        self.conv1 = SN(nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1))
        self.conv2 = SN(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1))
        self.conv3 = SN(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=1, padding=1))
        self.conv4 = SN(nn.Conv2d(base_ch * 4, 1, 4, stride=1, padding=1))

    def forward(self, source, target_or_gen):
        s_hf = haar_hf_concat(source)              # [B, 9, 128, 128]
        t_hf = haar_hf_concat(target_or_gen)       # [B, 9, 128, 128]
        x = torch.cat([s_hf, t_hf], dim=1)         # [B, 18, 128, 128]
        x = F.leaky_relu(self.conv1(x), 0.2)       # [B, 64, 64, 64]
        x = F.leaky_relu(self.conv2(x), 0.2)       # [B, 128, 32, 32]
        x = F.leaky_relu(self.conv3(x), 0.2)       # [B, 256, 31, 31]
        return self.conv4(x)                        # [B, 1, 30, 30]


class GaussianBottleneckHead(nn.Module):
    """VIB head after UNI fuse: h_fused (in_ch) -> mu/logvar (z_ch), then expand to dec_ch.

    KL is mean over B,C,H,W of 0.5*(mu^2 + sigma^2 - logvar - 1). Spec: C9 VIB encoder.md.
    """
    def __init__(self, in_ch=512, z_ch=64, dec_ch=512):
        super().__init__()
        self.z_ch = int(z_ch)
        self.pre = nn.Conv2d(in_ch, in_ch, 1)
        self.to_stat = nn.Conv2d(in_ch, 2 * self.z_ch, 1)
        self.expand = nn.Conv2d(self.z_ch, int(dec_ch), 1)

    def forward(self, h_fused, sample=True):
        h = F.gelu(self.pre(h_fused))
        mu, logvar = self.to_stat(h).chunk(2, dim=1)
        logvar = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        if sample and self.training:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        kl = 0.5 * (mu.pow(2) + std.pow(2) - logvar - 1.0).mean()
        z_dec = self.expand(z)
        return dict(z=z, z_dec=z_dec, mu=mu, std=std, logvar=logvar, kl=kl)


class C9Paired(nn.Module):
    """Full C9 paired-AE model: 2 encoders + fuse + 2 decoders. No skips.

    vib=True: HE→IHC only. Locked UNI fuse (768→512), then Gaussian z_ch bottle,
    expand to fuse_out_ch into Dec_IHC. No E_IHC / Fuse_IHC / Dec_HE.
    """
    def __init__(self, uni_module, taps=(13, 19), n_cls=1, n_reg=8,
                  use_grn_decoder=False, use_grn_encoder=False,
                  layer_scale_init=0.1, use_layer_scale=True,
                  use_layer_scale_encoder=False,
                  z_ch=512, computational_ch=512, fuse_out_ch=512,
                  expand_before_fuse=False, no_dilation=False,
                  vib=False, vib_z_ch=64, z_hw=128):
        super().__init__()
        self.vib = bool(vib)
        self.vib_z_ch = int(vib_z_ch)
        # Under VIB the encoder bottle stays locked 512; vib_z_ch is the Gaussian width.
        if self.vib:
            z_ch = int(computational_ch)
            expand_before_fuse = False
        self.z_ch = int(z_ch)
        self.computational_ch = int(computational_ch)
        self.fuse_out_ch = int(fuse_out_ch)
        self.expand_before_fuse = bool(expand_before_fuse)
        self.no_dilation = bool(no_dilation)
        self.z_hw, self.stem_strides, self.upsample_factor = derive_spatial(z_hw)
        enc_d = (1, 1, 1, 1) if self.no_dilation else (1, 2, 5, 1)
        dec_d = (1, 1, 1, 1, 1) if self.no_dilation else (2, 5, 1, 2, 5)
        self.uni_extractor = UNIFeatureExtractor(
            uni_module, taps=taps, target_size=224,
            n_cls=n_cls, n_reg=n_reg)
        # Encoder LS is independent of decoder LS. Default off: y = x + h_GRN(x).
        self.E_HE = C9Encoder(
            computational_ch=computational_ch, z_ch=z_ch,
            expand_before_fuse=expand_before_fuse, dilations=enc_d,
            use_grn=use_grn_encoder, layer_scale_init=layer_scale_init,
            use_layer_scale=use_layer_scale_encoder,
            stem_strides=self.stem_strides)
        fuse_in_ch = self.E_HE.out_ch
        self.Fuse_HE = UNIBottleneckFuse(
            z_ch=fuse_in_ch, out_ch=fuse_out_ch, taps=taps, target_hw=self.z_hw)
        self.Dec_IHC = C9Decoder(in_ch=fuse_out_ch, use_grn=use_grn_decoder,
                                layer_scale_init=layer_scale_init,
                                use_layer_scale=use_layer_scale, dilations=dec_d,
                                upsample_factor=self.upsample_factor)
        if self.vib:
            self.E_IHC = None
            self.Fuse_IHC = None
            self.Dec_HE = None
            self.vib_head = GaussianBottleneckHead(
                in_ch=fuse_out_ch, z_ch=self.vib_z_ch, dec_ch=fuse_out_ch)
        else:
            self.E_IHC = C9Encoder(
                computational_ch=computational_ch, z_ch=z_ch,
                expand_before_fuse=expand_before_fuse, dilations=enc_d,
                use_grn=use_grn_encoder, layer_scale_init=layer_scale_init,
                use_layer_scale=use_layer_scale_encoder,
                stem_strides=self.stem_strides)
            self.Fuse_IHC = UNIBottleneckFuse(
                z_ch=fuse_in_ch, out_ch=fuse_out_ch, taps=taps, target_hw=self.z_hw)
            self.Dec_HE = C9Decoder(in_ch=fuse_out_ch, use_grn=use_grn_decoder,
                                   layer_scale_init=layer_scale_init,
                                   use_layer_scale=use_layer_scale, dilations=dec_d,
                                   upsample_factor=self.upsample_factor)
            self.vib_head = None

    def encode(self, x, modality):
        if self.vib and modality != "HE":
            raise RuntimeError("VIB-C9 has no IHC encoder")
        z_raw = (self.E_HE if modality == "HE" else self.E_IHC)(x)
        with torch.no_grad():
            uni_feats = self.uni_extractor(x)
        fuse = self.Fuse_HE if modality == "HE" else self.Fuse_IHC
        return fuse(z_raw, uni_feats)

    def decode(self, z, modality):
        if self.vib and modality != "IHC":
            raise RuntimeError("VIB-C9 has no HE decoder")
        return (self.Dec_HE if modality == "HE" else self.Dec_IHC)(z)

    def forward_vib_h2i(self, HE, sample=True):
        """HE → fuse → Gauss(z_ch) → expand → Dec_IHC. No IHC encode / HE decode."""
        if not self.vib:
            raise RuntimeError("forward_vib_h2i requires vib=True")
        h_fused = self.encode(HE, "HE")
        vb = self.vib_head(h_fused, sample=sample)
        gen_IHC = self.Dec_IHC(vb["z_dec"])
        return dict(
            h_fused=h_fused,
            z_HE=vb["z"],
            z_dec=vb["z_dec"],
            mu=vb["mu"], std=vb["std"], logvar=vb["logvar"], kl=vb["kl"],
            gen_IHC=gen_IHC,
            # placeholders so shared logging does not KeyError
            z_IHC=vb["z"],
            rec_HE=gen_IHC, rec_IHC=gen_IHC, gen_HE=gen_IHC,
        )

    def forward_all_paths(self, HE, IHC):
        """4 forward paths: rec_HE, rec_IHC, gen_IHC (H2I main), gen_HE (I2H rev)."""
        if self.vib:
            return self.forward_vib_h2i(HE, sample=True)
        z_HE = self.encode(HE, "HE")
        z_IHC = self.encode(IHC, "IHC")
        rec_HE = self.decode(z_HE, "HE")
        rec_IHC = self.decode(z_IHC, "IHC")
        gen_IHC = self.decode(z_HE, "IHC")
        gen_HE = self.decode(z_IHC, "HE")
        return dict(z_HE=z_HE, z_IHC=z_IHC,
                     rec_HE=rec_HE, rec_IHC=rec_IHC,
                     gen_IHC=gen_IHC, gen_HE=gen_HE)

    def forward(self, HE, IHC=None, sample=True):
        """DDP hooks this. VIB ignores IHC and samples z unless sample=False."""
        if self.vib:
            return self.forward_vib_h2i(HE, sample=sample)
        if IHC is None:
            raise ValueError("IHC is required when vib=False")
        return self.forward_all_paths(HE, IHC)


# =============================================================================
# LOSS MODULES
# =============================================================================
class LatentPatchNCE(nn.Module):
    """Symmetric PatchNCE on fused bottleneck latents z_HE and z_IHC."""
    def __init__(self, n_patches=256, temperature=0.07):
        super().__init__()
        self.n_patches = n_patches
        self.T = temperature

    def _oneway_nce(self, q_full, k_full, ids):
        B, N, C = q_full.shape
        q_s = q_full[:, ids, :]
        k_s = k_full[:, ids, :]
        n = q_s.size(1)
        sim = torch.bmm(q_s, k_s.transpose(1, 2)) / self.T
        labels = torch.arange(n, device=q_s.device).unsqueeze(0).expand(B, n)
        return F.cross_entropy(sim.reshape(B * n, n), labels.reshape(-1))

    def forward(self, z_he, z_ihc):
        B, C, H, W = z_he.shape
        q_he = F.normalize(z_he.permute(0, 2, 3, 1).reshape(B, H * W, C), dim=-1)
        q_ihc = F.normalize(z_ihc.permute(0, 2, 3, 1).reshape(B, H * W, C), dim=-1)
        n = min(self.n_patches, H * W)
        ids = torch.randperm(H * W, device=q_he.device)[:n]
        return 0.5 * (self._oneway_nce(q_he, q_ihc, ids) + self._oneway_nce(q_ihc, q_he, ids))


class LatentPairNCE(nn.Module):
    """Matched HE/IHC tile as +, other tiles (in-batch + FIFO bank) as -.

    Locked L3 is spatial PatchNCE on one pair. This is the pair-level
    ablation: GAP-pool each fused z to one token, positive = same filename
    pair, negatives = other batch items plus a detached memory bank.
    Batch 2 only gives one in-batch negative; the bank is required.
    """

    def __init__(self, dim=512, bank_size=4096, temperature=0.07):
        super().__init__()
        self.T = temperature
        self.dim = int(dim)
        self.bank_size = int(bank_size)
        self.register_buffer("bank_he", torch.zeros(self.bank_size, self.dim))
        self.register_buffer("bank_ihc", torch.zeros(self.bank_size, self.dim))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("n_filled", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _pool(z):
        v = z.float().mean(dim=(2, 3))
        return F.normalize(v, dim=-1)

    def _enqueue(self, he, ihc):
        he_d = he.detach()
        ihc_d = ihc.detach()
        B = he_d.size(0)
        p = int(self.ptr.item())
        end = p + B
        if end <= self.bank_size:
            self.bank_he[p:end] = he_d
            self.bank_ihc[p:end] = ihc_d
        else:
            first = self.bank_size - p
            self.bank_he[p:] = he_d[:first]
            self.bank_ihc[p:] = ihc_d[:first]
            rest = B - first
            self.bank_he[:rest] = he_d[first:]
            self.bank_ihc[:rest] = ihc_d[first:]
        self.ptr.fill_(end % self.bank_size)
        self.n_filled.fill_(min(self.bank_size, int(self.n_filled.item()) + B))

    def _oneway(self, q, k_pos, k_bank, n_bank):
        logits = torch.matmul(q, k_pos.t()) / self.T
        if n_bank > 0:
            # Clone: enqueue inplaces the FIFO buffers. A view would be
            # versioned into autograd and blow up on the next backward
            # ([dim, B] saved from bank.t(), seen on step 2).
            bank = k_bank[:n_bank].detach().clone()
            logits = torch.cat(
                [logits, torch.matmul(q, bank.t()) / self.T], dim=1)
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(self, z_he, z_ihc):
        q = self._pool(z_he)
        k = self._pool(z_ihc)
        n_bank = int(self.n_filled.item())
        loss = 0.5 * (
            self._oneway(q, k, self.bank_ihc, n_bank)
            + self._oneway(k, q, self.bank_he, n_bank)
        )
        with torch.no_grad():
            self._enqueue(q, k)
        return loss

