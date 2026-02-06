# marmo3DPoseViewer
[Kaneko et al.](https://doi.org/10.1016/j.cub.2024.05.033),の論文と[marmo3Dpose](https://github.com/PrimatoModelling/marmo3Dpose)で用いられているpickleデータを3次元空間に描画する目的で制作されました。

## Install

Requirements:

* python >= 3.9
* numpy
* [pyqtgraph](https://github.com/pyqtgraph/pyqtgraph)
  * PyQt5
  * pyopengl
  * cupy (optional)
* pyav
* pims

```shell
pip install -r requirements.txt
```

## Quickstart
visualizemod_speed_stop_play_seekbar.pyの中の最後付近に下の行があります \
これを見たいディレクトリとsessionに変更してください
```python
    resultpath = '../marmo3Dpose/results/3d_v0p8_dark_fix_20'
    filename = 'dailylife_cj611_20230226_110000'
```

またはpicklefileを自分で書いてください
```python
    picklefile = '/home/user/marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_cj611_20230226_110000'
```
実行は（おそらく）これ
```python
python visualizemod_speed_stop_play_seekbar.py
```


## Citation

```bibtex
@article{KANEKO20242854,
title = {Deciphering social traits and pathophysiological conditions from natural behaviors in common marmosets},
journal = {Current Biology},
volume = {34},
number = {13},
pages = {2854-2867.e5},
year = {2024},
issn = {0960-9822},
doi = {https://doi.org/10.1016/j.cub.2024.05.033},
url = {https://www.sciencedirect.com/science/article/pii/S0960982224006766},
author = {Takaaki Kaneko and Jumpei Matsumoto and Wanyi Lu and Xincheng Zhao and Louie Richard Ueno-Nigh and Takao Oishi and Kei Kimura and Yukiko Otsuka and Andi Zheng and Kensuke Ikenaka and Kousuke Baba and Hideki Mochizuki and Hisao Nishijo and Ken-ichi Inoue and Masahiko Takada},
keywords = {markerless motion capture, pose estimation, artificial intelligence, natural behavior, data-driven analysis, food-sharing, mentalizing, hidden state, Parkinson’s disease, α-synuclein},
abstract = {Summary
Nonhuman primates (NHPs) are indispensable animal models by virtue of the continuity of behavioral repertoires across primates, including humans. However, behavioral assessment at the laboratory level has so far been limited. Employing the application of three-dimensional (3D) pose estimation and the optimal integration of subsequent analytic methodologies, we demonstrate that our artificial intelligence (AI)-based approach has successfully deciphered the ethological, cognitive, and pathological traits of common marmosets from their natural behaviors. By applying multiple deep neural networks trained with large-scale datasets, we established an evaluation system that could reconstruct and estimate the 3D poses of the marmosets, a small NHP that is suitable for analyzing complex natural behaviors in laboratory setups. We further developed downstream analytic methodologies to quantify a variety of behavioral parameters beyond motion kinematics. We revealed the distinct parental roles of male and female marmosets through automated detections of food-sharing behaviors using a spatial-temporal filter on 3D poses. Employing a recurrent neural network to analyze 3D pose time series data during social interactions, we additionally discovered that marmosets adjusted their behaviors based on others’ internal state, which is not directly observable but can be inferred from the sequence of others’ actions. Moreover, a fully unsupervised approach enabled us to detect progressively appearing symptomatic behaviors over a year in a Parkinson’s disease model. The high-throughput and versatile nature of an AI-driven approach to analyze natural behaviors will open a new avenue for neuroscience research dealing with big-data analyses of social and pathophysiological behaviors in NHPs.}
}
```


3DPoseViewerを参考に作成しました
https://github.com/Embracing/3DPoseViewer

```bibtex
@inproceedings{ci2023proactive,
  title={Proactive Multi-Camera Collaboration for 3D Human Pose Estimation},
  author={Hai Ci and Mickel Liu and Xuehai Pan and fangwei zhong and Yizhou Wang},
  booktitle={The Eleventh International Conference on Learning Representations },
  year={2023},
  url={https://openreview.net/forum?id=CPIy9TWFYBG}
}
```

## License

Apache License, Version 2.0.
