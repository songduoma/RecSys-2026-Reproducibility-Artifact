<h2 align="center">
Are We Really Making Progress in Group Recommendation?<br>
Unmasking the Tie-Breaking Illusion
</h2>

<p align="center">
<b>Song-Duo Ma, Pu-Jen Cheng</b>
</p>

<p align="center">
National Taiwan University, Taipei, Taiwan
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3773078.3831853" target="_blank">
    <img src="https://img.shields.io/badge/RecSys-2026-blue.svg?style=flat-square" alt="RecSys 2026">
  </a>
  <a href="https://arxiv.org/abs/xxxx.xxxxx" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg?style=flat-square">
  </a>

</p>



## Overview

This repository contains the code and data for our paper on evaluation bias in group recommendation. We show that an extra sigmoid applied before the BPR loss can compress scores and create many exact ties, making HR@K and NDCG@K highly sensitive to deterministic tie-breaking. We revisit five recent methods and their baselines on CAMRa2011 and Mafengwo, introduce tie-aware evaluation that computes the exact expected metrics under uniform random tie-breaking, and show that several reported gains shrink substantially under this protocol. We further study the role of the extra sigmoid as implicit margin smoothing and explore temperature-scaled BPR as a simple mitigation.

## What's in this repository

- Full training/evaluation code for **ConsRec**, **AlignGroup**, **DHMAE**, **ITR**, **DGGVAE**, and the baselines **AGREE**, **GroupIM**, **HCR**, **HyperGroup**, **HHGR**, **CubeRec**, each patched with our tie-aware evaluator.
- The tie-aware HR@K / NDCG@K implementation (exact expectation under uniform random tie-breaking, plus top-score tie statistics).
- A simple code switch to restore the original pre-BPR sigmoid behavior for controlled comparison.

```
TieAwareGroupRec/
├── WWW2023ConsRec/     ConsRec (WWW '23), incl. temperature-scaled BPR (τ-BPR)
├── AlignGroup/          AlignGroup (CIKM '24)
├── DHMAE/               DHMAE (SIGIR '24)
├── ITR/                 ITR (NeurIPS '24)
├── DGGVAE/               DGGVAE (TOIS '26)
├── Baseline/            AGREE, GroupIM, HCR, HyperGroup, HHGR, CubeRec
└── environment.yml
```

Each method folder is self-contained (data, model, and training scripts) and can be run independently of the others.


## Requirements

- Python 3.9
- PyTorch 2.5.1 (CUDA 12.1)
- `torch_geometric`, `numpy`, `scipy`, `scikit-learn`, `tensorboardX`

```bash
conda env create -f environment.yml
conda activate TieAwareGroupRec
```

All experiments in the paper were conducted on Ubuntu 22.04 using single-precision (FP32) computation. We retain FP32 throughout because numerical precision interacts directly with the tie-inflation phenomenon studied in this work. environment.yml pins Python to 3.9 rather than a specific patch version for portability across Conda channels.



## Running the experiments

<details>
<summary><b>ConsRec (WWW '23)</b></summary>

```bash
cd WWW2023ConsRec

# Mafengwo
for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR \
    --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 \
    --epoch=200 --tau=1 --seed=$s
done

# CAMRa2011
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --predictor=DOT --loss_type=BPR \
    --learning_rate=0.001 --device=cuda:0 --num_negatives=2 --layers=2 \
    --epoch=30 --tau=1 --seed=$s
done
```
</details>

<details>
<summary><b>AlignGroup (CIKM '24)</b></summary>

```bash
cd AlignGroup

# Mafengwo (temp=0.2)
for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --device=cuda:0 --seed=$s
done

# CAMRa2011 (temp=0.8)
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --device=cuda:0 --seed=$s
done
```
</details>

<details>
<summary><b>DGGVAE (TOIS '26)</b></summary>

```bash
cd DGGVAE

# Mafengwo defaults: k=50, temp=0.2
for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --device=cuda:0 --seed=$s
done

# CAMRa2011 defaults: k=60, temp=0.4 (set in main.py before running)
for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --device=cuda:0 --seed=$s
done
```
</details>

<details>
<summary><b>DHMAE (SIGIR '24)</b></summary>

```bash
cd DHMAE

for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --num_negatives=6 --num_enc_layers=1 \
    --num_dec_layers=3 --sce_alpha=1 --drop_ratio=0.0 --epoch=30 \
    --seed=$s --device=cuda:0
done

for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --num_negatives=10 --num_enc_layers=2 \
    --num_dec_layers=3 --sce_alpha=2 --drop_ratio=0.1 --epoch=200 \
    --seed=$s --device=cuda:0
done
```

`run.sh` gives the base hyperparameter templates; we average over three seeds for all paper-reported numbers.
</details>

<details>
<summary><b>ITR (NeurIPS '24)</b></summary>

```bash
cd ITR

for s in 0 1 2; do
  python -u main.py --dataset=CAMRa2011 --predictor=DOT --loss_type=BPR \
    --learning_rate=0.001 --device=cuda:0 --num_negatives=2 --layers=2 \
    --epoch=30 --seed=$s > output_camra2011_revised_seed${s}.log 2>&1
done

for s in 0 1 2; do
  python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR \
    --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 \
    --epoch=2000 --seed=$s > output_mafengwo_revised_seed${s}.log 2>&1
done
```

ITR writes to stdout by default; we redirect to `.log` files for bookkeeping.
</details>

<details>
<summary><b>Baselines (AGREE, GroupIM, HyperGroup, HHGR, CubeRec, HCR)</b></summary>

```bash
SEEDS=(0 1 2)

cd Baseline/AGREE
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

cd ../GroupIM
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

cd ../HyperGroup
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

cd ../HHGR
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 > log_rerun/camra2011_seed${s}.log 2>&1
done

cd ../CubeRec
for s in "${SEEDS[@]}"; do
  python -u main.py --dataset=Mafengwo  --seed=$s --device=cuda:0 --epoch=100 > log_rerun/mafengwo_seed${s}.log 2>&1
  python -u main.py --dataset=CAMRa2011 --seed=$s --device=cuda:0 --epoch=30  > log_rerun/camra2011_seed${s}.log 2>&1
done
```

`HCR` reads its hyperparameters from `Baseline/HCR/config.py` rather than argparse — switch datasets by editing `self.path` before running `python main.py`.
</details>

<details>
<summary><b>Temperature-scaled BPR sweep</b></summary>

```bash
cd WWW2023ConsRec
for tau in 1 2 4 8 16 32 64; do
  for seed in 0 1 2; do
    python -u main.py --dataset=Mafengwo --predictor=MLP --loss_type=BPR \
      --learning_rate=0.0001 --device=cuda:0 --num_negatives=8 --layers=3 \
      --epoch=200 --tau=$tau --seed=$seed
  done
done
```

</details>

## Original vs. revised (no-extra-sigmoid) implementations

The code in this repository defaults to the **revised** path (extra sigmoid removed) for every affected method. To reproduce the original, buggy behavior — e.g. to regenerate `log_original` from scratch — re-enable the sigmoid in:

`AlignGroup/model.py`, `WWW2023ConsRec/model.py`, `ITR/model.py`, `DGGVAE/model.py`, `DHMAE/model.py`, `Baseline/AGREE/model.py`, `Baseline/HCR/model.py`, `Baseline/HyperGroup/model.py`.

(`GroupIM`, `HHGR`, and `CubeRec` never had this issue and are unaffected by the switch.) 

## Notes

- A few scripts declare list-valued hyperparameters via `argparse(..., type=list, ...)`; the dataset-specific values used for the paper are set as code defaults rather than passed on the command line.

## Acknowledgments

This work builds directly on the official implementations of [ConsRec](https://github.com/FDUDSDE/WWW2023ConsRec), [AlignGroup](https://github.com/Jinfeng-Xu/AlignGroup), [DHMAE](https://github.com/ICharlotteI/DHMAE), [ITR](https://github.com/yueliu1999/ITR), [DGGVAE](https://github.com/Jinfeng-Xu/DGGVAE), and the [group recommendation baselines](https://github.com/FDUDSDE/WWW2023GroupRecBaselines) released alongside ConsRec. We thank the respective authors for making their code public. This work was supported by the National Science and Technology Council (NSTC), Taiwan, under Grant NSTC 115-2634-F-001-006.

## Citation

```bibtex
@inproceedings{ma2026tieaware,
  title     = {Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion},
  author    = {Ma, Song-Duo and Cheng, Pu-Jen},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems (RecSys '26)},
  year      = {2026},
  address   = {Minneapolis, MN, USA},
  publisher = {ACM},
  doi       = {10.1145/3773078.3831853}
}
```

If you use the reproduced baseline or method implementations, please also cite the corresponding original papers (ConsRec, AlignGroup, DHMAE, ITR, DGGVAE, and the respective baselines).
