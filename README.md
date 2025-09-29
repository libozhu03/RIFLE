<div align="center">
  <p align="center">
    <img src="asserts/figs/logo.png" width="400px">
  </p>
</div>

# RIFLE: Removal of Image Flicker-Banding via Latent Diffusion Enhancement

[LiBo Zhu](https://github.com/LiBoZhu030073), [Zihan Zhou](), [Xiaoyang Liu](), [Weihang Zhang](), [Keyu Shi](), [Yifan Fu]() and [Yulun Zhang](http://yulunzhang.com/)  
**"RIFLE: Removal of Image Flicker-Banding via Latent Diffusion Enhancement", arxiv 2025**

[![page](https://img.shields.io/badge/Project-Page-blue?logo=github)](https://labor555.github.io/PassionSR_Web/)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2411.17106)
[![supp](https://img.shields.io/badge/Supplementary_material-Paper-orange.svg)](https://github.com/user-attachments/files/22590138/compress_arxiv_supp_Libo_Zhu__ICLR_26_Debanding.pdf)
[![releases](https://img.shields.io/github/downloads/libozhu03/RIFLE/total.svg)](https://github.com/libozhu03/RIFLE/releases)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=libozhu03.RIFLE&right_color=violet)](https://github.com/libozhu03/RIFLE)
[![GitHub Stars](https://img.shields.io/github/stars/libozhu03/RIFLE?style=social)](https://github.com/libozhu03/RIFLE)

---

## 📚 Table of Contents

- [🔥 News](#-news)
- [📘 Abstract](#-abstract)
- [📝 Structure Overview](#-structure-overview)
- [⚙️ Installation](#️-installation)
- [📥 Download Pretrained Models and Datasets](#-download-pretrained-models-and-datasets)
- [🧪 Inference](#-inference)
- [🔎 Results](#-results)
- [📝 Acknowledgements](#-acknowledgements)
- [📌 Citation](#-citation)

---

## 🔥 News
- **[2025-09-29]** Create repository.

### ⭐⭐⭐ If RIFLE is helpful to your projects, please help star this repo. Thanks!
---

## 📘 Abstract

>  Capturing screens is now routine in our everyday lives. But the photographs of emissive displays are often influenced by the flicker-banding (FB), which is alternating bright–dark stripes that arise from temporal aliasing between a camera’s rolling-shutter readout and the display’s brightness modulation. Unlike moiré degradation, which has been extensively studied, the FB remains underexplored despite its frequent and severe impact on readability and perceived quality. We formulate FB removal as a dedicated restoration task and introduce Removal of Image Flicker-Banding via Latent Diffusion Enhancement, RIFLE, a diffusion-based framework designed to remove FB while preserving fine details. We propose the flicker-banding prior estimator (FPE) that predicts key banding attributes and injects it into the restoration network. Additionally, Masked Loss (ML) is proposed to concentrate supervision on banded regions without sacrificing global fidelity. To overcome data scarcity, we provide a simulation pipeline that synthesizes FB in the luminance domain with stochastic jitter in banding angle, banding spacing, and banding width. Feathered boundaries and sensor noise are also applied for a more realistic simulation. For evaluation, we collect a paired real-world FB dataset with pixel-aligned banding-free references captured via long exposure. Across quantitative metrics and visual comparisons on our real-world dataset, RIFLE consistently outperforms recent image reconstruction baselines from mild to severe flicker-banding. To the best of our knowledge, it is the first work to research the simulation and removal of FB. Our work establishes a great foundation for subsequent research in both the dataset construction and the removal model design.

---

## 📝 Structure Overview
<p align="center">
  <img src="asserts/figs/simulate_01.jpg" width="800px">
</p>


<p align="center">
  <img src="asserts/figs/structure_01.jpg" width="800px">
</p>

|                      RealFlicker                      |                      GT                      |                   LQ                     |            MAT             | InvSR   |            PiSA-SR            |             Step1X          |           RIFLE      |
| :------------------------------------------: | :------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: |:---------------------------------------------: |:---------------------------------------------: |
| <img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/annotated_base/GT/Banding03_2_y0804_x0210_1.png" height=110> | <img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/GT/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> | <img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/LQ/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> | <img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/MAT/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> | <img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/InvSR/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> |<img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/PiSA-SR/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> |<img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/Step1X/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> |<img src="asserts/figs/visual_comp/img19_BASE0_0_500_500_BOXEs330_320_50_100/patches/RIFLE/Banding03_2_y0804_x0210_1_BASE_x0_y0_w500_h500_crop1_x330_y320_w50_h100.png" height=110> |
| <img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/annotated_base/GT/Banding15_2_y0954_x0692_1.png" height=110> | <img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/GT/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> | <img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/LQ/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> | <img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/MAT/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> | <img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/InvSR/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> |<img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/PiSA-SR/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> |<img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/Step1X/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> |<img src="asserts/figs/visual_comp/img80_BASE0_0_500_500_BOXEs80_170_50_100/patches/RIFLE/Banding15_2_y0954_x0692_1_BASE_x0_y0_w500_h500_crop1_x80_y170_w50_h100.png" height=110> |



</details>

---



## ⚙️ Installation
TBD

---

## 📥 Download Pretrained Models and Datasets
TBD

---

## 🧪 Inference
TBD

---

## <a name="-results"></a> 🔎 Results
RIFLE significantly out-performs previous methods, which are fintuned with our simulated datasets. 



<details>
<summary> 📊 Quantitative comparisons in Table 1 of the main paper (click to expand)</summary>

<p align="center">
  <img width="900" src="asserts/figs/results.png">
</p>
</details>

<details>
<summary> 🖼 Visual comparison in Figure 6 of the main paper (click to expand)
</summary>

<p align="center">
  <img width="900" src="asserts/figs/visual.png">
</p>
</details>

---

## 📝 Acknowledgements
We would like to thank the developers and maintainers of [Stable Diffusion](https://github.com/Stability-AI/StableDiffusion), [Diffusers](https://github.com/huggingface/diffusers), and [PiSA-SR](https://github.com/csslc/PiSA-SR) for their open-source contributions, which have greatly facilitated our research and development.

This project is supported in part by the Shanghai Jiao Tong University Artificial Intelligence Institute.

We also thank our collaborators and contributors for their valuable feedback and technical discussions.

## 📌 Citation

```bibtex

```