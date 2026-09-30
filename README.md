# Thesis H&E-to-IHC models and Chapter 4 evaluation

Model definitions and the scripts that fill the Chapter 4 tables for H&E-to-IHC translation.

https://github.com/augustdua/thesis-h2i-evaluation

Tiles: [augustander/her2match-full](https://huggingface.co/datasets/augustander/her2match-full). Checkpoints: [augustander/ch4-operating-ckpts](https://huggingface.co/augustander/ch4-operating-ckpts). Point `configs/chapter4.json` at the local copies. Split sizes are 11,610 train, 3,582 val, and 5,980 test. Layout is `HE/{train,val,test}` and `IHC/{train,val,test}`.

## Layout

- `exemplar/`: MatchFormer shallow-64 (`matchformer_pair.py`, `matchformer_shallow64.py`) and the matcher evals (`eval_k4.py`, `eval_baselines.py`).
- `direct_translators/`: C9 paired network (`c9_paired.py`: encoders, UNI fusion, decoders, PatchNCE, PairNCE, critics) and the translator evals.
- `vib/`: VIB entry (`network.py` builds `C9Paired(vib=True)` with `GaussianBottleneckHead`) and `score_vib.py`.
- `bridge/`: variance-exploding bridge U-Net (`unet.py`).
- `src/`: shared metrics. Linear CKA is in `src/cka.py`. DAB-L1 for saved images is `StainDeconvDAB` in `src/dab.py`. Frequency DAB is in `src/frequency.py`. Exemplar patch DAB is `dab_hed_ch2` in `exemplar/matchformer_pair.py`.
- `configs/`: paths and fixed settings. `configs/train_dab010.txt` is the 1,867-name DAB-enriched list for the Constant donor. `configs/hf_study_stems.json` is the 240-tile frequency list.
- `scripts/run_chapter4.py`: local runner for the eval scripts.
- `outputs/`: generated JSON and images (gitignored).

## Run

```bash
python -u scripts/run_chapter4.py list
python -u scripts/run_chapter4.py cka-rank -- --only a12
python -u scripts/run_chapter4.py frequency
python -u scripts/run_chapter4.py dab-l1
python -u scripts/run_chapter4.py vib
python -u scripts/run_chapter4.py exemplar-k4
python -u scripts/run_chapter4.py exemplar-baselines
```

Images are resized in the scripts: bilinear 1024 to 512 for the exemplar, bicubic to 256 for the translator metrics.

Exemplar checkpoints: K=4 is the shallow-64 file at epoch 197 (`exemplar_k4_ckpt`). K=10 and K=20 were saved every 25 epochs, so the files on disk are epoch 200 (`exemplar_k10_ckpt`, `exemplar_k20_ckpt`). Donor mosaics (4,096 donors) are `donor_he_grouped_512.npy` and `donor_ihc_grouped_512.npy` in `donor_dir`.

Fixed settings in `configs/chapter4.json`: seed 42, 400 tiles for CKA and spatial rank, DAB threshold 0.3, tissue optical density 0.15, 4,096 donors. The frequency study treats a tile as DAB-positive when the real IHC brown fraction is at least 0.05. The Constant donor is the lowest mean patch error on `configs/train_dab010.txt`.

The CKA script builds UNI with `pretrained=False` and then loads the checkpoint state dict. Pass a UNI module into `C9Paired`.

VIB weights are downloaded with the other checkpoints. The frequency table uses the 240-tile validation subset `vib_c64_b1e-4` when that folder is set. The fused-panel checkpoint name is `full_vib_c9_z64` at step 15500.

## Which script fills which table

| Chapter quantity | Script |
| --- | --- |
| Tables 4.18 and 4.19, learned K=4: selected error, top-4, exemplar DAB-L1 | `exemplar/eval_k4.py` |
| Tables 4.18 and 4.19, other rows: Oracle, Random, Constant, H&E nearest neighbour, learned K=10 and K=20 at epoch 200 | `exemplar/eval_baselines.py` |
| Table 4.4 linear CKA and Table 4.6 spatial effective rank (400 validation tiles, seed 42) | `direct_translators/cka_spatial_rank.py --only a12` |
| Wavelet and half-Nyquist shares for the 64 and 32 cross-decode grids | `direct_translators/cka_spatial_rank.py --only a4` |
| Table 4.7 brown fraction and DAB-L1, and Table 4.9 half-Nyquist and wavelet-band ratios | `direct_translators/hf_study.py` |
| Tissue-mask DAB-L1 and DAB-positive DAB-L1 on saved translator images | `direct_translators/dab_l1.py` |
| DAB-L1 on saved VIB images | `vib/score_vib.py` |
| Bootstrap 95% interval on per-tile means | `src/bootstrap.py` (`bootstrap_mean_ci`) |

`exemplar/eval_baselines.py` records the printed K=4 figures and scores the other rows. Patch error is mean absolute DAB plus mean absolute RGB on each 8x8 patch, against the library of 4,096 donors. Top-4 means the selected donor is among the four lowest-error donors.

`hf_study.py` reports brown fraction and DAB-L1 for the wavelet low-pass. The UNI cosine column of Table 4.7 is separate from that script.

## Requirements

`requirements.txt` lists `torch>=2.4.0` and `torchvision>=0.19.0`. The evaluation environment used `torch==2.4.1`, `torchvision==0.19.1`, `timm==1.0.11`, and `lpips==0.1.4`.
