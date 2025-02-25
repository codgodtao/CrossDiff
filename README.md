# Accepted by IEEE Trans ON Image Processing!!!
https://arxiv.org/abs/2401.05153

# update file in config

# training stage1 for Cross-predictive
python sr.py
python sr_pan2ms.py

# training stage2 for fusionhead
python fusion.py #reduced resolution
python fusion_full.py #full resolution


# Citation
'''
Y. Xing, L. Qu, S. Zhang, K. Zhang, Y. Zhang and L. Bruzzone, "CrossDiff: Exploring Self-SupervisedRepresentation of Pansharpening via Cross-Predictive Diffusion Model," in IEEE Transactions on Image Processing, vol. 33, pp. 5496-5509, 2024, doi: 10.1109/TIP.2024.3461476. keywords: {Pansharpening;Diffusion models;Feature extraction;Spatial resolution;Training;Noise reduction;Transformers;Image fusion;pansharpening;self-supervised learning;denoising diffusion probabilistic model},
'''
