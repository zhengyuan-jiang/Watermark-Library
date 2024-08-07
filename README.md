# Certifiably-Robust-Image-Watermark
This code is the official implementation of our ECCV'24 paper: Certifiably-Robust-Image-Watermark [Paper](https://arxiv.org/abs/2407.04086).


## Preparation

1. Clone this repo from the GitHub.

```
git clone https://github.com/zhengyuan-jiang/Watermark-Library.git
```		

2. Setup environment.

All experiments are run on a single RTX-6000 with 24GB GPU memory.

```
pip install -r requirements.txt
```	

3. Download checkpoint files.

You can download standard.pth [here](https://drive.google.com/file/d/1FazyK9XtWR05Y8c1565bve-1rayueC3b/view?usp=sharing) and adversarial.pth [here](https://drive.google.com/file/d/1AG-ZoB6w1Z6eV7AlpeT5Su7_cUqTgRIO/view?usp=drive_link). Move them into the `checkpoint` folder.

4. Download non-AI-generated image testing set.

You can download the testing set [here](https://drive.google.com/file/d/1pNHGW94UbFcabvxN8QXNRSxTCqu7C-NJ/view?usp=sharing). You can also use use your own datasets or use your model by modifying `network.py`.


## Evaluate certified robustness

Note that you should set `num_noise` to 10000 (which is consistent with the paper) if you wish to get the same results. Generally, with a larger `num_noise`, the estimation of certified robustness will be more accurate, but it will also take more time.

Evaluate certified robustness of multi-class, multi-label, and regression based smoothing methods:

```
python3 compare_smoothing_method.py
```

Evaluate certified robustness of standard training and adversarial training:

```
python3 compare_training_strategy.py
```

Evaluate certified robustness of different detection threshold $\tau$:

```
python3 compare_tau.py
```

Evaluate certified robustness of different size of randomized noises:

```
python3 compare_sigma.py
```


## Citation

If you find our work useful for your research, please consider citing the paper
```
@inproceedings{jiang2024certifiably,
  title={Certifiably Robust Image Watermark},
  author={Jiang, Zhengyuan and Guo, Moyang and Hu, Yuepeng and Jia, Jinyuan and Gong, Neil Zhenqiang},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

For our other interesting watermarking works, please refer as follows:

[WEvade](https://github.com/zhengyuan-jiang/WEvade) (white-box and black-box attacks to image watermarks)
```
@inproceedings{jiang2023evading,
  title={Evading watermark based detection of AI-generated content},
  author={Jiang, Zhengyuan and Zhang, Jinghuai and Gong, Neil Zhenqiang},
  booktitle={ACM Conference on Computer and Communications Security (CCS)},
  year={2023}
}
```

[Watermark-based attribution](https://arxiv.org/abs/2404.04254) with theoretical guarantees 
```
@article{jiang2024watermark,
  title={Watermark-based Detection and Attribution of AI-Generated Content},
  author={Jiang, Zhengyuan and Guo, Moyang and Hu, Yuepeng and Gong, Neil Zhenqiang},
  journal={arXiv preprint arXiv:2404.04254},
  year={2024}
}
```
