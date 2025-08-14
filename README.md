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

Here are qualitative results of RelayFormer on manipulated videos.  
All videos are directly playable below.

<table>
  <tr>
    <td align="center">
      <video src="assets/blackswan.mp4" controls autoplay loop muted style="width:100%; max-width:320px;"></video>
      <br>Demo 1
    </td>
    <td align="center">
      <video src="assets/breakdance-flare.mp4" controls autoplay loop muted style="width:100%; max-width:320px;"></video>
      <br>Demo 2
    </td>
    <td align="center">
      <video src="breakdance.mp4" controls autoplay loop muted style="width:100%; max-width:320px;"></video>
      <br>Demo 3
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="elephant.mp4" controls autoplay loop muted style="width:100%; max-width:320px;"></video>
      <br>Demo 4
    </td>
    <td align="center">
      <video src="motocross-jump.mp4" controls autoplay loop muted style="width:100%; max-width:320px;"></video>
      <br>Demo 5
    </td>
    <td align="center">
      <video src="assets/dance-twirl.mp4" controls autoplay loop muted style="width:100%; max-width:320px;"></video>
      <br>Demo 6
    </td>
  </tr>
</table>



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
  author={Huang, Wen and Yang, Jiarui and Dai, Tao and others},
  journal={arXiv preprint arXiv:2508.09459},
  year={2025}
}
```

---

## 🙏 Acknowledgements

We thank the authors of [IMDLBench](https://github.com/IMDLBench) and other open-source projects that made this work possible.
