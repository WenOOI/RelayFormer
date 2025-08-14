# RelayFormer

[![Paper](https://img.shields.io/badge/arXiv-2508.09459-b31b1b.svg)](https://arxiv.org/abs/2508.09459)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)]()

**RelayFormer: A Unified Local–Global Attention Framework for Scalable Image & Video Manipulation Localization**  
Official PyTorch implementation of our paper:  
> Wen Huang, Jiarui Yang, Mengdi Li, et al.  
> _RelayFormer: A Unified Local–Global Attention Framework for Scalable Image and Video Manipulation Localization_  
> [arXiv:2508.09459](https://arxiv.org/abs/2508.09459)

---

## 📌 Introduction
RelayFormer is a modular architecture designed for detecting manipulated regions in both images and videos at any resolution. It achieves **state-of-the-art accuracy** with high computational efficiency, and shows strong robustness against common perturbations such as blur, noise, and JPEG compression.

Key components:
- **Local Unit Construction** – Splits large images or long videos into scalable processing units without interpolation or excessive padding.
- **Global–Local Relay Attention (GLoRA)** – Introduces a small set of _Global Relay Tokens_ (GRTs) to enable global context exchange using minimal LoRA-style modifications.
- **Query-based Mask Decoder** – Decodes masks for an entire video in one pass using the first frame’s queries, achieving **one-shot inference**.

---

## 🎥 Demo

### Video Results
https://user-images.githubusercontent.com/your-demo-video.mp4

### Image Results
| Input Image | Ground Truth | RelayFormer Prediction |
|-------------|--------------|------------------------|
| ![](images/input1.jpg) | ![](images/gt1.jpg) | ![](images/pred1.jpg) |
| ![](images/input2.jpg) | ![](images/gt2.jpg) | ![](images/pred2.jpg) |

---

## 📂 Repository Structure

```

RelayFormer/
├── README.md
├── LICENSE
├── images/             # Images for README/demo
├── videos/             # Demo videos
├── src/
│   ├── dataset/
│   ├── models/
│   ├── utils/
│   ├── inference.py
│   └── train.py
├── checkpoints/        # Pretrained weights (coming soon)
├── requirements.txt
└── setup.py

````

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/RelayFormer.git
cd RelayFormer
pip install -r requirements.txt
````

**Dependencies:**

* Python >= 3.8
* PyTorch >= 2.0
* torchvision
* OpenCV, NumPy
* tqdm, Pillow

---

## 🚀 Usage

### 1. Inference (Coming Soon)

```bash
python src/inference.py \
  --input path/to/image_or_video \
  --checkpoint checkpoints/relayformer.pth \
  --output output_folder
```

### 2. Training (Planned Release)

```bash
python src/train.py \
  --dataset path/to/dataset \
  --epochs 200 \
  --batch-size 16 \
  --learning-rate 3e-4
```

---

## 📅 Roadmap / TODO

* [ ] **Release pretrained weights** (Relay-ViT / Relay-Seg)
* [ ] **Publish inference code** with sample scripts & demo
* [ ] **Publish training code** with full configurations
* [ ] Add documentation for datasets & evaluation

---

## 📄 Citation

If you use RelayFormer in your research, please cite:

```bibtex
@article{huang2025relayformer,
  title={RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization},
  author={Huang, Wen and Yang, Jiarui and Li, Mengdi and others},
  journal={arXiv preprint arXiv:2508.09459},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We thank the authors of [IMDLBench](https://github.com/IMDLBench) and other open-source projects that made this work possible.
