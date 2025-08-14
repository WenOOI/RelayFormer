# RelayFormer

[![Paper](https://img.shields.io/badge/arXiv-2508.09459-b31b1b.svg)](https://arxiv.org/abs/2508.09459)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)]()

**RelayFormer: A Unified Local–Global Attention Framework for Scalable Image & Video Manipulation Localization**  
Official PyTorch implementation of our paper:  
> Wen Huang, Jiarui Yang, Tao Dai, et al.  
> _RelayFormer: A Unified Local–Global Attention Framework for Scalable Image and Video Manipulation Localization_  
> [arXiv:2508.09459](https://arxiv.org/abs/2508.09459)

---

## 📌 Introduction
RelayFormer is a modular architecture designed for detecting manipulated regions in both images and videos at dynamic resolution. It achieves **state-of-the-art accuracy** with high computational efficiency.

---

## 🎥 Demo

### Video Results

Here are qualitative results of RelayFormer on manipulated videos, where specific objects have been removed from the original footage.
The red mask regions indicate the areas predicted by the model as having been removed or altered from the original content.

<table>
  <tr>
    <td align="center">
      <img src="assets/blackswan.gif" width="320"><br>Demo 1
    </td>
    <td align="center">
      <img src="assets/breakdance-flare.gif" width="320"><br>Demo 2
    </td>
    <td align="center">
      <img src="assets/breakdance.gif" width="320"><br>Demo 3
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/elephant.gif" width="320"><br>Demo 4
    </td>
    <td align="center">
      <img src="assets/motocross-jump.gif" width="320"><br>Demo 5
    </td>
    <td align="center">
      <img src="assets/dance-twirl.gif" width="320"><br>Demo 6
    </td>
  </tr>
</table>


## 📅 Roadmap / TODO

* [ ] **Release pretrained weights** (Relay-ViT)
* [ ] **Publish inference code** with sample scripts & demo
* [ ] **Publish training code** with full configurations
* [ ] Add documentation for datasets & evaluation

---

## 📄 Citation

If you use RelayFormer in your research, please cite:

```bibtex
@article{huang2025relayformer,
  title={RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization},
  author={Huang, Wen and Yang, Jiarui and Dai, Tao and others},
  journal={arXiv preprint arXiv:2508.09459},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We thank the authors of [IMDLBench](https://github.com/IMDLBench) and other open-source projects that made this work possible.
