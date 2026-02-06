# marmo3DPoseViewer
[Kaneko et al.](https://doi.org/10.1016/j.cub.2024.05.033),の論文で用いられているpickleデータを3次元空間に描画する目的で制作されました。

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
