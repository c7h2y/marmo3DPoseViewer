import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
)
from PyQt5.QtCore import QTimer
import pyqtgraph.opengl as gl
import pyqtgraph as pg


class Pose3DViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Pose Viewer (auto-play)")

        # ここで読み込んだデータを保持する変数を用意
        self.kp3d = None       # shape = (n_animal, n_frame, n_kp, 3)
        self.current_frame = 0
        self.n_frame = 0
        self.n_animal = 0
        cm = plt.get_cmap('hsv', 36)
        self.kp_con = [
            {'name':'0_2','color':cm(27), 'bodypart':(0,2)},
            {'name':'0_1','color':cm(31),'bodypart':(0,1)},
            {'name':'2_4','color':cm(29), 'bodypart':(2,4)},
            {'name':'1_3','color':cm(33),'bodypart':(1,3)},
            #{'name':'0_4','color':cm(29), 'bodypart':(0,4)},
            #{'name':'0_3','color':cm(33),'bodypart':(0,3)},
            {'name':'6_8','color':cm(5),'bodypart':(6,8)},
            {'name':'5_7','color':cm(10),'bodypart':(5,7)},
            {'name':'8_10','color':cm(7),'bodypart':(8,10)},
            {'name':'7_9','color':cm(12),'bodypart':(7,9)},
            {'name':'12_14','color':cm(16),'bodypart':(12,14)},
            {'name':'11_13','color':cm(22),'bodypart':(11,13)},
            {'name':'14_16','color':cm(18),'bodypart':(14,16)},
            {'name':'13_15','color':cm(24),'bodypart':(13,15)},
            {'name':'18_17','color':cm(1),'bodypart':(18,17)},
            {'name':'0_18','color':cm(1),'bodypart':(0,18)},
            {'name':'18_6','color':cm(2),'bodypart':(18,6)},
            {'name':'18_5','color':cm(3),'bodypart':(18,5)},
            {'name':'17_12','color':cm(14),'bodypart':(17,12)},
            {'name':'17_11','color':cm(20),'bodypart':(17,11)}
            ]

        # ウィジェット配置
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # pyqtgraph の OpenGL 3D ウィジェット
        self.glview = gl.GLViewWidget()
        layout.addWidget(self.glview)

        # 軸や床をセットアップ
        self._init_scene_items()

        # 再生を進めるボタン (任意)
        self.btn_next = QPushButton("Next frame (manual)")
        layout.addWidget(self.btn_next)
        self.btn_next.clicked.connect(self.next_frame)

        # タイマーを使って自動でフレームを進める
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        # 例: 30ミリ秒ごと(約33fps)にフレームを更新
        self.timer.start(30)

    def _init_scene_items(self):
        # 床
        g = gl.GLGridItem()
        g.scale(50, 50, 1)  # 格子
        self.glview.addItem(g)

        # 軸 (簡易的にX:赤, Y:緑, Z:青)
        axis_len = 100
        x_ax = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_len, 0, 0]]),
                                 color=(1, 0, 0, 1), width=2)
        y_ax = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_len, 0]]),
                                 color=(0, 1, 0, 1), width=2)
        z_ax = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_len]]),
                                 color=(0, 0, 1, 1), width=2)
        self.glview.addItem(x_ax)
        self.glview.addItem(y_ax)
        self.glview.addItem(z_ax)
        
        self.glview.opts['distance'] = 1000

    def shift_kp3d_to_origin(self, kp_index=0):
        """
        kp3d: shape = (n_animal, n_frame, n_kp, 3)
        kp_index: 原点とみなすキーのインデックス (0ならkeypoint 0)
        
        各 animal の最初のフレーム (frame=0) の keypoint=kp_index の座標を
        (0,0,0) に合わせるように、kp3d 全体を平行移動する。
        """
        n_animal, n_frame, n_kp, _ = self.kp3d.shape

        # フレーム0, 関節kp_index の座標をオフセットとする
        offset = self.kp3d[0, 0, kp_index].copy()  # shape=(3,)
          
        for i_animal in range(n_animal):
            # 全フレーム・全キーポイントをこのoffsetぶん引く
            self.kp3d[i_animal] -= offset  # ブロードキャストされる

        # return kp3d  # in-placeでもよいが、一応戻り値として返す


    def load_pickle(self, pickle_file):
        """
        pickleファイルを読み込み、
        data['kp3d'] (n_animal, n_frame, n_kp, 3) を self.kp3d にセット。
        """
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # たとえば "kp3d" というキーで形状 (n_animal, n_frame, n_kp, 3) を想定
        self.kp3d = data['kp3d']  # shape=(n_animal, n_frame, n_kp, 3)
        self.shift_kp3d_to_origin()
        self.rotate_3d_kp()
        self.n_animal, self.n_frame, self.n_kp, _ = self.kp3d.shape
        self.current_frame = 0

        # 初回フレームを描画
        self.draw_frame(0)
        
    def rotate_3d_kp(self, deg=270):
        """
        kp3d: shape = (n_kp, 3)
            各行に 3D座標 [x, y, z] が入っているとする
        deg: 回転角度(度数法) 
            ここでは x軸を中心に +deg 度だけ回転するとする

        return:
            rotated_kp3d: shape = (n_kp, 3)
                        回転後の座標
        """
        # 角度をラジアンに変換
        theta = math.radians(deg)

        # x軸回りの回転行列 (右手系, 反時計回り +theta)
        #   R_x(theta) =
        #   [[ 1, 0,      0],
        #    [ 0, cos, -sin],
        #    [ 0, sin,  cos]]
        R = np.array([
            [1,              0,               0],
            [0,  math.cos(theta), -math.sin(theta)],
            [0,  math.sin(theta),  math.cos(theta)]
        ], dtype=np.float32)

        # (n_kp,3) x (3,3) -> (n_kp,3)
        self.kp3d=self.kp3d @ R.T

    def draw_frame(self, frame_idx):
        """
        frame_idx番目の (n_animal, n_kp, 3) を3D描画。
        いったん前のフレームで描いた点やラインは削除し、再度描画。
        """
        # 既存のLinePlotItemなどを一度削除（床や軸以外を消す）
        for item in self.glview.items[:]:
            if not isinstance(item, gl.GLGridItem):
                # 軸はGLLinePlotItemだが、先頭3つ（床を含め4つ）以外を削除するなど工夫
                if item not in (self.glview.items[1], self.glview.items[2], self.glview.items[3]):
                    self.glview.removeItem(item)

        # (n_animal, n_kp, 3) のうち frame_idx 番を取り出す -> shape=(n_animal, n_kp, 3)
        for i_animal in range(self.n_animal):
            pose3d = self.kp3d[i_animal, frame_idx]  # shape=(n_kp, 3)

            # もし全ゼロならスキップ(出現していない等)
            if not np.any(pose3d):
                continue

            # 適当な描画 (キーポイントをラインで繋ぐにはスケルトン情報が必要)
            self._draw_pose(pose3d, color=(1, 0, 0, 1))

        self.current_frame = frame_idx

    def _draw_pose(self, pose3d, color=(1,1,1,1)):
        """
        pose3d: shape = (n_kp, 3)
        3D座標 [x, y, z] を持つキーポイントが n_kp 個なら、
        pose3d[i] = [x_i, y_i, z_i]

        self.kp_con: 上記のようなリスト (global変数 kp_con をクラス内で保持している想定)
        self.glview: pyqtgraph.opengl.GLViewWidget など
        """

        for seg in self.kp_con:
            j1, j2 = seg['bodypart']   # (int, int)
            color_rgba = seg['color']  # (R, G, B, A) in [0,1]

            # インデックス範囲外チェック (念のため)
            if j1 >= len(pose3d) or j2 >= len(pose3d):
                continue

            p1 = pose3d[j1]  # 例: [x1, y1, z1]
            p2 = pose3d[j2]  # 例: [x2, y2, z2]

            # 未検出の場合はスキップ例
            # if not np.any(p1) or not np.any(p2):
            #     continue

            linepos = np.array([p1, p2], dtype=np.float32)  # shape=(2,3)

            # GLLinePlotItem で2点を結ぶ線分を描画
            plt_item = gl.GLLinePlotItem(
                pos=linepos,
                color=color_rgba,  # (R,G,B,A) in 0~1
                width=3,
                antialias=True,
                mode='lines'       # 2点1本の線
            )
            self.glview.addItem(plt_item)
    
    def _clear_pose(self):
        """前フレームで描画したスケルトン等を削除"""
        remove_list = []
        for item in self.glview.items:
            # 床や軸などを除外し、スケルトン線だけ remove
            # たとえば linepos.shape == (2,3) とか name をチェックするなど方法は色々
            # ここでは "width=3" などの条件で特定する例:
            if isinstance(item, gl.GLLinePlotItem) and item.width == 3:
                remove_list.append(item)

        for it in remove_list:
            self.glview.removeItem(it)

    
    def next_frame(self):
        """
        ボタン or QTimer から呼ばれ、次のフレームに進む
        """
        if self.kp3d is None:
            return
        new_idx = (self.current_frame + 1) % self.n_frame
        self._clear_pose()
        self.draw_frame(new_idx)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Pose3DViewer()
    viewer.resize(1200, 800)
    viewer.show()

    # pickleファイルを読み込む
    # ここを実際のファイルパスに合わせてください
    resultpath='../marmo3Dpose/pose'
    filename='foodcomp_cj753_cj813_20240111_132306'
    pickleFile = os.path.join(resultpath,filename,'kp3d.pickle')  # 例
    viewer.load_pickle(pickleFile)

    sys.exit(app.exec_())