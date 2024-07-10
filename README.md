# Certifiably-Robust-Image-Watermark
This code is the official implementation of our ECCV'24 paper: Certifiably-Robust-Image-Watermark [Paper](https://arxiv.org/abs/2407.04086).

## Preparation

1. Clone this repo from the GitHub.
	
		git clone https://github.com/zhengyuan-jiang/Watermark-Library.git

2. Setup environment.



3. Download checkpoint files.

You can download standard.pth [Here](https://drive.google.com/file/d/1FazyK9XtWR05Y8c1565bve-1rayueC3b/view?usp=sharing) and adversarial.pth [Here](https://drive.google.com/file/d/1AG-ZoB6w1Z6eV7AlpeT5Su7_cUqTgRIO/view?usp=drive_link). Move them into checkpoint folder.

4. Download non-AI-generated image testing set.

You can download the testing set [Here](https://drive.google.com/file/d/1pNHGW94UbFcabvxN8QXNRSxTCqu7C-NJ/view?usp=sharing).


## Evaluate certified robustness

Note that you should set num_noise to be 10000 (which is consistent to the paper) if you wish to get the same results. Generally, given a larger num_noise, the estimation of certified robustness will be more accurate, but will also cost more time.

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
