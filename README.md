# Chapter 4 evaluation

Evaluation code for the Chapter 4 H&E-to-IHC numbers: exemplar retrieval, direct translators, and the variational information bottleneck (VIB). Dataset tiles and model checkpoints are separate downloads. This repository does not contain them, and `outputs/` is not committed.

Repository: https://github.com/augustdua/thesis-h2i-evaluation

## Layout

- `exemplar/`: MatchFormer shallow-64 retrieval. Oracle, selected error, top-4, Random, Constant, H&E nearest neighbour, and learned K=4, K=10, K=20.
- `direct_translators/`: DAB-L1, linear CKA, spatial effective rank, wavelet energy, and power above half the Nyquist frequency.
- `vib/`: the same DAB-L1 on saved VIB images, plus a list of checkpoints and image sets that were not retained.
- `src/`: the metric functions those scripts import. Linear CKA is defined once, in `src/cka.py`.
- `configs/`: fixed hyperparameters and local paths. `configs/train_dab010.txt` is the 1,867-name DAB-enriched list used to pick the Constant donor. `configs/hf_study_stems.json` is the 240-tile list for the frequency study. Neither file is a score dump.
- `scripts/run_chapter4.py`: one local runner. It does not start Modal.
- `outputs/`: gitignored. JSON and any generated images go here.

## What you download

Set paths in `configs/chapter4.json`. Empty strings mean "not set".

HER2 tiles, paired JPEGs, laid out as `HE/{train,val,test}` and `IHC/{train,val,test}` under `her2_root`. The chapter used the release `augustander/her2match-full`. Expected split sizes are 11,610 train, 3,582 val, and 5,980 test. Images are resized in the scripts (bilinear 1024 to 512 for the exemplar; bicubic to 256 for the translator metrics).

Operating translator checkpoints were released as `augustander/ch4-operating-ckpts`. This package does not download them and does not read API tokens.

Exemplar extras, not in those dataset cards:

- K=4 matcher: the shallow-64 checkpoint at epoch 197 (`exemplar_k4_ckpt`). `exemplar/eval_k4.py` is that eval (`_scratch/eval_shallow64_tables_418_419.py`).
- K=10 and K=20: `exemplar_k10_ckpt` and `exemplar_k20_ckpt`. Checkpoints were written every 25 epochs, so the saved file is epoch 200. Epochs 192 (K=10) and 188 (K=20) were not saved.
- Donor mosaics, 4,096 donors: `donor_he_grouped_512.npy` and `donor_ihc_grouped_512.npy` in `donor_dir`.

`bbdm_root` is only required for CKA, spatial effective rank, and any decode that builds `C9Paired`. Those classes stay in `bbdm_BCI/train_c9_paired.py` of a BBDM checkout, together with `bbdm_BCI/src_ds_her2_thunder.py`. They are not copied here: `train_c9_paired.py` reads a local token file on import. Image metrics on saved PNGs (DAB-L1, wavelet, half-Nyquist) do not need it. The CKA script builds UNI with `pretrained=False` and then loads the checkpoint state dict, which is the procedure in `compute_thesis_diag_a12_a4.py`.

Fixed settings recorded in `configs/chapter4.json`: seed 42, 400 tiles for CKA and spatial rank, DAB threshold 0.3, tissue optical density 0.15, 4,096 donors. The frequency study marks a tile DAB-positive when the real IHC brown fraction is at least 0.05. The Constant donor is the lowest mean patch error on `configs/train_dab010.txt`, not on all 11,610 training tiles.

## Which script fills which quantity

| Chapter quantity | Script |
| --- | --- |
| K=4 selected error, top-4, DAB-L1 (the matcher eval behind 0.2790 / 0.5477 and 9.98% / 7.93%) | `exemplar/eval_k4.py` |
| Oracle, Random, Constant, H&E nearest neighbour, learned K=10 and K=20 at epoch 200 | `exemplar/eval_baselines.py` |
| Linear CKA and spatial effective rank (Tables 4.5 and 4.6) | `direct_translators/cka_spatial_rank.py` (`--only a12`) |
| Wavelet low-pass on real IHC (Table 4.7: brown fraction and DAB-L1) and half-Nyquist power (Table 4.9) | `direct_translators/hf_study.py` |
| DAB-L1 inside the H&E tissue mask, and on target DAB-positive pixels (Table 4.4 style) | `direct_translators/dab_l1.py` |
| The same DAB-L1 for VIB image folders you still have | `vib/score_vib.py` |
| Bootstrap 95% interval on per-tile means | `src/bootstrap.py` (`bootstrap_mean_ci`) |

`exemplar/eval_baselines.py` does not recompute the learned K=4 row. It records the printed K=4 figures and scores the other rows, including K=10 and K=20. Patch error is mean absolute DAB plus mean absolute RGB on each 8x8 patch, against the fixed library of 4,096 donors. Top-4 means the selected donor is among the four lowest-error donors.

Exemplar DAB is `dab_hed_ch2` in `exemplar/matchformer_pair.py` (3-stain HED, channel 2). Translator image DAB-L1 is `StainDeconvDAB` in `src/dab.py` (Ruifrok H and DAB only). The frequency study uses the numpy Ruifrok DAB in `src/frequency.py`. These are the three definitions the original scripts already used.

## VIB checkpoints that were not retained

No VIB weights are committed. Full validation and test images for the spatial VIB grids (32, 64, and 256) were not kept. Full validation and test images for VIB at 128 and beta 1e-4 were not kept either; the frequency table used a 240-tile validation subset (`vib_c64_b1e-4`) when those PNGs are supplied. Other beta values were not kept as full image sets. The fused-panel checkpoint name was `full_vib_c9_z64` at step 15500. Supply that file yourself if you recompute the representation row.

## Not wired

- UNI cosine on the wavelet-low-pass tiles (the UNI column of Table 4.7). `hf_study.py` computes brown fraction and DAB-L1 only. A separate UNI script for that column was not in the frequency study, so this package does not add one.
- The Modal test farm (`eval_translator_test_modal.py`) is not included. It reads a token and launches remote jobs.
- The 10-subset FID piece of the old bootstrap diagnostic is not included. Only the per-tile mean resampling loop is in `src/bootstrap.py`.

## Requirements

`requirements.txt` follows the BBDM training pins that already existed (`torch>=2.4.0`, `torchvision>=0.19.0`) and leaves the other packages unpinned. The remote test image happened to use `torch==2.4.1`, `torchvision==0.19.1`, `timm==1.0.11`, and `lpips==0.1.4`.
