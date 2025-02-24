## Scalable Diffusion Models with Transformers (DiT)<br><sub>Official PyTorch Implementation</sub>

## Setup

```bash
conda env create -n DiT python=3.12
conda activate DiT
conda install pip
pip install -r requirements.txt
```


## Sampling [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/wpeebles/DiT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/facebookresearch/DiT/blob/main/run_DiT.ipynb)
![More DiT samples](visuals/sample_grid_1.png)

**Pre-trained DiT checkpoints.** You can sample from our pre-trained DiT models with [`sample.py`](sample.py). Weights for our pre-trained DiT model will be 
automatically downloaded depending on the model you use. The script has various arguments to switch between the 256x256
and 512x512 models, adjust sampling steps, change the classifier-free guidance scale, etc. For example, to sample from
our 512x512 DiT-XL/2 model, you can use:

```bash
python sample.py --image-size 512 --seed 1
```

For convenience, our pre-trained DiT models can be downloaded directly here as well:

| DiT Model     | Image Resolution | FID-50K | Inception Score | Gflops | 
|---------------|------------------|---------|-----------------|--------|
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) | 256x256          | 2.27    | 278.24          | 119    |
| [XL/2](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt) | 512x512          | 3.04    | 240.82          | 525    |


**Custom DiT checkpoints.** If you've trained a new DiT model with [`train.py`](train.py) (see [below](#training-dit)), you can add the `--ckpt`
argument to use your own checkpoint instead. For example, to sample from the EMA weights of a custom 
256x256 DiT-L/4 model, run:

```bash
python sample.py --model DiT-L/4 --image-size 256 --ckpt /path/to/model.pt
```


## Training DiT

We provide a training script for DiT in [`train.py`](train.py). This script can be used to train class-conditional 
DiT models, but it can be easily modified to support other types of conditioning. To launch DiT-XL/2 (256x256) training with `N` GPUs on 
one node:

```bash
torchrun  --nproc_per_node=1 --master-port 29502 train.py --model DiT-XL/2 --data-path ../../datasets/latent-imagenet-256/data_index.json
```

### PyTorch Training Results

We've trained DiT-XL/2 and DiT-B/4 models from scratch with the PyTorch training script
to verify that it reproduces the original JAX results up to several hundred thousand training iterations. Across our experiments, the PyTorch-trained models give 
similar (and sometimes slightly better) results compared to the JAX-trained models up to reasonable random variation. Some data points:

| DiT Model  | Train Steps | FID-50K<br> (JAX Training) | FID-50K<br> (PyTorch Training) | PyTorch Global Training Seed |
|------------|-------------|----------------------------|--------------------------------|------------------------------|
| XL/2       | 400K        | 19.5                       | **18.1**                       | 42                           |
| B/4        | 400K        | **68.4**                   | 68.9                           | 42                           |
| B/4        | 400K        | 68.4                       | **68.3**                       | 100                          |

These models were trained at 256x256 resolution; we used 8x A100s to train XL/2 and 4x A100s to train B/4. Note that FID 
here is computed with 250 DDPM sampling steps, with the `mse` VAE decoder and without guidance (`cfg-scale=1`). 

**TF32 Note (important for A100 users).** When we ran the above tests, TF32 matmuls were disabled per PyTorch's defaults. 
We've enabled them at the top of `train.py` and `sample.py` because it makes training and sampling way way way faster on 
A100s (and should for other Ampere GPUs too), but note that the use of TF32 may lead to some differences compared to 
the above results.

### Enhancements
Training (and sampling) could likely be sped-up significantly by:
- [ ] using [Flash Attention](https://github.com/HazyResearch/flash-attention) in the DiT model
- [ ] using `torch.compile` in PyTorch 2.0

Basic features that would be nice to add:
- [ ] Monitor FID and other metrics
- [ ] Generate and save samples from the EMA model periodically
- [ ] Resume training from a checkpoint
- [ ] AMP/bfloat16 support

**🔥 Feature Update** Check out this repository at https://github.com/chuanyangjin/fast-DiT to preview a selection of training speed acceleration and memory saving features including gradient checkpointing, mixed precision training and pre-extrated VAE features. With these advancements, we have achieved a training speed of 0.84 steps/sec for DiT-XL/2 using just a single A100 GPU.

## Evaluation (FID, Inception Score, etc.)

We include a [`sample_ddp.py`](sample_ddp.py) script which samples a large number of images from a DiT model in parallel. This script 
generates a folder of samples as well as a `.npz` file which can be directly used with [ADM's TensorFlow
evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and
other metrics. For example, to sample 50K images from our pre-trained DiT-XL/2 model over `N` GPUs, run:

```bash
torchrun --nnodes=1 --nproc_per_node=N sample_ddp.py --model DiT-XL/2 --num-fid-samples 50000
```

There are several additional options; see [`sample_ddp.py`](sample_ddp.py) for details. 


## Differences from JAX

Our models were originally trained in JAX on TPUs. The weights in this repo are ported directly from the JAX models. 
There may be minor differences in results stemming from sampling with different floating point precisions. We re-evaluated 
our ported PyTorch weights at FP32, and they actually perform marginally better than sampling in JAX (2.21 FID 
versus 2.27 in the paper).


## BibTeX

```bibtex
@article{Peebles2022DiT,
  title={Scalable Diffusion Models with Transformers},
  author={William Peebles and Saining Xie},
  year={2022},
  journal={arXiv preprint arXiv:2212.09748},
}
```


## Acknowledgments
We thank Kaiming He, Ronghang Hu, Alexander Berg, Shoubhik Debnath, Tim Brooks, Ilija Radosavovic and Tete Xiao for helpful discussions. 
William Peebles is supported by the NSF Graduate Research Fellowship.

This codebase borrows from OpenAI's diffusion repos, most notably [ADM](https://github.com/openai/guided-diffusion).


## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.
