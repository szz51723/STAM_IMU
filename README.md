# STAM
This repository is the code of "An Adaptive IMU Bias Inference Framework Based on Deep Learning for Cross-scene Calibration". 

## License
The code is shared only for **research purposes**, and cannot be used for any commercial purposes.

Authors: Zhuoer Wang, Baohan Shi, Chenming Zhang, Xiaowen Zhu, Hongjuan Zhang*, Bijun Li.

## Abstract
The inertial measurement unit (IMU) is a core sensor in modern intelligent systems, and its attitude estimationpose estimation accuracy is critical for applications such as unmanned aerial vehicles (UAV) and autonomous driving. IMU measurement errors can be broadly de-composed into sensor hardware-induced offset biases and environment-induced dynamic noise. Existing approaches typically focus on offset biases but often fail to handle dynamic noise arising from environmental changes, which leads to marked performance degradation when models are deployed across different scenes. To address the challenge, a deep learning-based adaptive IMU calibration framework is proposed, which predicts the combined error arising from the coupling of offset biases and dynamic noise. The framework comprises a multimodal feature extraction unit for scene understanding, a hidden-state layer that supports lightweight online updates, and a state-space module for temporal error modeling. By updating only a subset of hidden-state parameters through a compact online learning procedure rather than retraining the full model, the proposed design enables rapid adaptation to dynamic errors and achieves high-precision, cross-scene IMU attitude estima-tionpose estimation. Experiments on the EuRoC and TUM public benchmarks as well as on our own dataset demonstrate that the method preserves high attitude-estimation accuracy under environment transfer and exhibits strong engineering applicability. Moreover, the approach substantially reduces model-update cost, offering practical value for cross-scene IMU deployment. Relevant code files are publicly available on GitHub.

## Acknowledgment
This research was supported by the State Key Program of National Natural Science Foundation of China (52332010). The numerical calculations in this article have been done on the supercomputing system in the Supercomputing Center of Wuhan University. The authors also acknowledge all editors and reviewers for their suggestions.

Copyright © 2025 State Key Laboratory of Information Engineering in Surveying, Mapping and Remote Sensing, University of Wuhan.  
For questions, contact wangzhuoer@whu.edu.cn


