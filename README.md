# Accepted by IEEE Trans ON Image Processing!!!
https://arxiv.org/abs/2401.05153

# update file in config

# training stage1 for Cross-predictive
python sr.py
python sr_pan2ms.py

# training stage2 for fusionhead
python fusion.py #reduced resolution
python fusion_full.py #full resolution


# pretrained checkpoint is comming soon