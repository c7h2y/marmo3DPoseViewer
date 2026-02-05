import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout, QLabel
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg


class Pose3DViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Pose Viewer (auto-play with controls)")

        # ここで読み込んだデータを保持する変数を用意
        self.kp3d = None       # shape = (n_animal, n_frame, n_kp, 3)
        self.current_frame = 0
        self.n_frame = 0
        self.n_animal = 0

        cm = plt.get_cmap('hsv', 36)
        self.kp_con = [
            {'name': '0_2', 'color': cm(27), 'bodypart': (0, 2)},
            {'name': '0_1', 'color': cm(31), 'bodypart': (0, 1)},
            {'name': '2_4', 'color': cm(29), 'bodypart': (2, 4)},
            {'name': '1_3', 'color': cm(33), 'bodypart': (1, 3)},
            {'name': '6_8', 'color': cm(5),  'bodypart': (6, 8)},
            {'name': '5_7', 'color': cm(10), 'bodypart': (5, 7)},
            {'name': '8_10','color': cm(7),  'bodypart': (8, 10)},
            {'name': '7_9', 'color': cm(12), 'bodypart': (7, 9)},
            {'name': '12_14','color': cm(16),'bodypart': (12, 14)},
            {'name': '11_13','color': cm(22),'bodypart': (11, 13)},
            {'name': '14_16','color': cm(18),'bodypart': (14, 16)},
            {'name': '13_15','color': cm(24),'bodypart': (13, 15)},
            {'name': '18_17','color': cm(1), 'bodypart': (18, 17)},
            {'name': '17_6','color': cm(2),  'bodypart': (17, 6)},
            {'name': '17_5','color': cm(3),  'bodypart': (17, 5)},
            {'name': '17_12','color': cm(14),'bodypart': (17, 12)},
            {'name': '17_11','color': cm(20),'bodypart': (17, 11)}
        ]

        # ウィジェット配置
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # pyqtgraph の OpenGL 3D ウィジェット
        self.glview = gl.GLViewWidget()
        main_layout.addWidget(self.glview)

        # 軸や床をセットアップ
        self._init_scene_items()

        # ---- 再生・停止ボタン ----
        btn_layout = QHBoxLayout()
        main_layout.addLayout(btn_layout)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_video)
        btn_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_video)
        btn_layout.addWidget(self.btn_stop)

        self.btn_next = QPushButton("Next frame")
        self.btn_next.clicked.connect(self.next_frame)
        btn_layout.addWidget(self.btn_next)

        # ---- 再生速度スライダー ----
        # 例: 1..200 ms/frame (1msなら1000fps、200msなら5fps)
        speed_layout = QHBoxLayout()
        main_layout.addLayout(speed_layout)
        speed_label = QLabel("Speed (ms/frame):")
        speed_layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 200)  # 1~200 ms
        self.speed_slider.setValue(33)      # デフォルト33ms ~約30fps
        self.speed_slider.valueChanged.connect(self.change_speed)
        speed_layout.addWidget(self.speed_slider)

        # ---- シークバー (フレーム移動) ----
        seek_layout = QHBoxLayout()
        main_layout.addLayout(seek_layout)
        seek_label = QLabel("Seek frame:")
        seek_layout.addWidget(seek_label)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)  # 後で n_frame ロード後にセット
        self.seek_slider.valueChanged.connect(self.seek_frame)
        seek_layout.addWidget(self.seek_slider)

        # フレーム番号表示ラベルを追加
        self.frame_label = QLabel("Frame: 0 / 0")
        seek_layout.addWidget(self.frame_label)

        # タイマーを使って自動でフレームを進める (初期は約30fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(33)  # 33ms => ~30fps
        self.timer.start()

    def _init_scene_items(self):
        """床や座標軸を配置し、カメラのdistanceを調整"""
        # 床
        self.grid_item = gl.GLGridItem()
        self.grid_item.scale(50, 50, 1)  # 格子
        self.glview.addItem(self.grid_item)

        # x,y,z 軸
        axis_len = 100
        self.x_ax = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [axis_len, 0, 0]]),
            color=(1, 0, 0, 1), width=2
        )
        self.y_ax = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, axis_len, 0]]),
            color=(0, 1, 0, 1), width=2
        )
        self.z_ax = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, axis_len]]),
            color=(0, 0, 1, 1), width=2
        )

        self.glview.addItem(self.x_ax)
        self.glview.addItem(self.y_ax)
        self.glview.addItem(self.z_ax)

        # 「恒常アイテム」をリストにまとめる (インスタンス属性)
        self.permanent_items = [
            self.grid_item,
            self.x_ax,
            self.y_ax,
            self.z_ax
        ]

        # カメラを少し遠くに
        self.glview.opts['distance'] = 1000


    def shift_kp3d_to_origin(self, kp_index=0):
        """
        kp3d: shape = (n_animal, n_frame, n_kp, 3)
        kp_index: 原点とみなすキーのインデックス (0ならkeypoint 0)
        
        各 animal の最初のフレーム (frame=0) の keypoint=kp_index の座標を
        (0,0,0) に合わせるように、kp3d 全体を平行移動する。
        """
        n_animal, n_frame, n_kp, _ = self.kp3d.shape

        offset = self.kp3d[0, 0, kp_index].copy()
        for i_animal in range(n_animal):
            self.kp3d[i_animal] -= offset  # in-place

    def rotate_3d_kp(self, deg=270):
        """
        x軸まわりに +deg 度回転
        """
        theta = math.radians(deg)
        R = np.array([
            [1,              0,               0],
            [0,  math.cos(theta), -math.sin(theta)],
            [0,  math.sin(theta),  math.cos(theta)]
        ], dtype=np.float32)
        # (n_animal, n_frame, n_kp, 3)
        # reshape -> (n_animal * n_frame * n_kp, 3) にして行列積
        shape_ = self.kp3d.shape
        reshaped = self.kp3d.reshape(-1, 3)
        rotated = reshaped @ R.T
        self.kp3d = rotated.reshape(shape_)

    def load_pickle(self, pickle_file):
        """
        pickleファイルを読み込み、self.kp3d にセットし
        原点シフト & 回転 を行う。
        """
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        self.kp3d = data['kp3d']  # (n_animal, n_frame, n_kp, 3)
        self.shift_kp3d_to_origin()
        self.rotate_3d_kp()
        self.n_animal, self.n_frame, self.n_kp, _ = self.kp3d.shape
        self.current_frame = 0

        # シークバーの最大値を設定
        self.seek_slider.setRange(0, self.n_frame - 1)

        # フレーム番号表示を更新
        self.frame_label.setText(f"Frame: {self.current_frame + 1} / {self.n_frame}")

        # 初回フレームを描画
        self.draw_frame(0)

    def draw_frame(self, frame_idx):
        """frame_idx番目の (n_animal, n_kp, 3) を3D描画。"""
        # 1) 既存の描画アイテムを全て調べ、permanent_items以外を削除
        for item in self.glview.items[:]:
            if item not in self.permanent_items:
                self.glview.removeItem(item)

        # 2) スケルトン等を新たに描画
        for i_animal in range(self.n_animal):
            pose3d = self.kp3d[i_animal, frame_idx]
            if not np.any(pose3d):
                continue
            self._draw_pose(pose3d)

        self.current_frame = frame_idx
        # フレーム番号表示を更新
        self.frame_label.setText(f"Frame: {self.current_frame + 1} / {self.n_frame}")

    def _draw_pose(self, pose3d):
        for seg in self.kp_con:
            j1, j2 = seg['bodypart']
            color_rgba = seg['color']  # (R,G,B,A) in 0~1
            if j1 >= len(pose3d) or j2 >= len(pose3d):
                continue
            p1 = pose3d[j1]
            p2 = pose3d[j2]
            linepos = np.array([p1, p2], dtype=np.float32)
            plt_item = gl.GLLinePlotItem(
                pos=linepos,
                color=color_rgba,
                width=3,
                antialias=True,
                mode='lines'
            )
            self.glview.addItem(plt_item)

    def _clear_pose(self):
        remove_list = []
        for item in self.glview.items:
            if item not in self.permanent_items:
                remove_list.append(item)
        for it in remove_list:
            self.glview.removeItem(it)

    # ---------------------------
    # コントロール系のメソッド
    # ---------------------------
    def play_video(self):
        """Play ボタンが押されたら timer を再開"""
        # speed_sliderの値を再反映
        interval_ms = self.speed_slider.value()
        self.timer.setInterval(interval_ms)
        self.timer.start()

    def stop_video(self):
        """Stop ボタンが押されたら timer を停止"""
        self.timer.stop()

    def next_frame(self):
        """1フレーム進める (タイマー / ボタン / etc. で呼ばれる)"""
        if self.kp3d is None or self.n_frame == 0:
            return
        new_idx = (self.current_frame + 1) % self.n_frame
        self._clear_pose()
        self.draw_frame(new_idx)

        # シークバー側も同期する
        self.seek_slider.blockSignals(True)  # シグナルループ防止
        self.seek_slider.setValue(new_idx)
        self.seek_slider.blockSignals(False)

    def change_speed(self, value):
        """
        speed_slider の値(1~200) が変化したときに呼ばれる。
        ここでは ms/frame として timer のインターバルに設定する
        """
        self.timer.setInterval(value)

    def seek_frame(self, frame_index):
        """
        シークバーが移動したときに呼ばれる。
        直接フレーム idx を指定して描画
        """
        if self.kp3d is None:
            return
        if 0 <= frame_index < self.n_frame:
            self._clear_pose()
            self.draw_frame(frame_index)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Pose3DViewer()
    viewer.resize(1200, 800)
    viewer.show()

    # pickleファイルを読み込む（パスを適宜書き換えてください）
    resultpath = '../marmo3Dpose/results/3d_v0p8_dark_fix_20'
    filename = 'dailylife_cj611_20230226_110000'
    pickleFile = os.path.join(resultpath, filename, 'kp3d_fxdJointLen.pickle')
    viewer.load_pickle(pickleFile)

    sys.exit(app.exec_())
