import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QSlider, QHBoxLayout, QLabel, QSpinBox
)
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph.opengl as gl
import pyqtgraph as pg

class Pose3DViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Pose Viewer (with face orientation)")

        # pose data: (n_animal, n_frame, n_kp, 3)
        self.kp3d = None
        self.current_frame = 0
        self.n_frame = 0
        self.n_animal = 0

        cm = plt.get_cmap('hsv', 36)
        # existing connections
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
        # specify keypoint indices for eyes/ears and nose
        self.LEFT_EYE = 1
        self.RIGHT_EYE = 2
        self.LEFT_EAR = 3
        self.RIGHT_EAR = 4
        self.NOSE = 0

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
        self._init_controls(main_layout)
        
                # タイマーを使って自動でフレームを進める (初期は約30fps)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(33)  # 33ms => ~30fps
        self.timer.start()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            # toggle play/stop
            if self.timer.isActive():
                self.stop_video()
            else:
                self.play_video()
        else:
            super().keyPressEvent(event)
    
    def _init_scene_items(self):
        self.grid = gl.GLGridItem(); self.grid.scale(50,50,1); self.glview.addItem(self.grid)
        axis_len = 100
        self.x_ax = gl.GLLinePlotItem(pos=np.array([[0,0,0],[axis_len,0,0]]), color=(1,0,0,1), width=2)
        self.y_ax = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,axis_len,0]]), color=(0,1,0,1), width=2)
        self.z_ax = gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,axis_len]]), color=(0,0,1,1), width=2)
        for item in (self.grid, self.x_ax, self.y_ax, self.z_ax): self.glview.addItem(item)
        self.permanent_items = [self.grid, self.x_ax, self.y_ax, self.z_ax]
        self.glview.opts['distance'] = 500

    def _init_controls(self, parent):
        # Buttons
        btn_layout = QHBoxLayout(); parent.addLayout(btn_layout)
        for name, slot in [('Play', self.play_video), ('Stop', self.stop_video), ('Next', self.next_frame)]:
            b = QPushButton(name); b.clicked.connect(slot); btn_layout.addWidget(b)

        # Speed slider
        sp_layout = QHBoxLayout(); parent.addLayout(sp_layout)
        sp_layout.addWidget(QLabel('Speed (ms/frame):'))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1,200)
        self.speed_slider.setValue(33)
        self.speed_slider.valueChanged.connect(self.change_speed)
        sp_layout.addWidget(self.speed_slider)

        # Seek and frame input
        sk_layout = QHBoxLayout(); parent.addLayout(sk_layout)
        sk_layout.addWidget(QLabel('Frame:'))
        # Seek slider styling slimmer
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #ccc; }"
            "QSlider::handle:horizontal { width: 12px; margin: -4px 0; }"
        )
        self.seek_slider.setRange(0,0)
        self.seek_slider.valueChanged.connect(self.seek_frame)
        sk_layout.addWidget(self.seek_slider)
        # Direct frame entry
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1,1)
        self.frame_spin.valueChanged.connect(lambda v: self.seek_frame(v-1))
        sk_layout.addWidget(self.frame_spin)
    
    def goto_frame(self, idx):
        self.current_frame = idx
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(idx)
        self.seek_slider.blockSignals(False)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(idx+1)
        self.frame_spin.blockSignals(False)
        self.draw_frame(idx)
    
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
    
    def load_pickle(self, path):
        with open(path,'rb') as f:
            data = pickle.load(f)
        self.kp3d = data['kp3d']
        self.shift_kp3d_to_origin()
        self.rotate_3d_kp()
        
        # center/orient
        # offset = self.kp3d[0,0,0].copy()
        # self.kp3d -= offset
        # rotate if needed...
        self.n_frame = self.kp3d.shape[1]
        self.n_animal = self.kp3d.shape[0]
        self.seek_slider.setRange(0, self.n_frame-1)
        self.frame_spin.setRange(1, self.n_frame)
        self.goto_frame(0)

    def compute_camera_axes(self, v, roll):
        """
        From forward vector v and roll, compute orthonormal axes.
        """
        # Choose global up (z-axis)
        up_g = np.array([0,0,1], dtype=np.float32)
        # Right axis
        right = np.cross(up_g, v)
        if np.linalg.norm(right) < 1e-6:  # forward nearly vertical
            up_g = np.array([0,1,0], dtype=np.float32)
            right = np.cross(up_g, v)
        right /= np.linalg.norm(right)
        # Camera up before roll
        up_cam = np.cross(v, right)
        # Apply roll around v axis
        theta = math.radians(roll)
        R = np.array([
            [math.cos(theta) + v[0]*v[0]*(1-math.cos(theta)),      v[0]*v[1]*(1-math.cos(theta))-v[2]*math.sin(theta), v[0]*v[2]*(1-math.cos(theta))+v[1]*math.sin(theta)],
            [v[1]*v[0]*(1-math.cos(theta))+v[2]*math.sin(theta), math.cos(theta)+v[1]*v[1]*(1-math.cos(theta)),      v[1]*v[2]*(1-math.cos(theta))-v[0]*math.sin(theta)],
            [v[2]*v[0]*(1-math.cos(theta))-v[1]*math.sin(theta), v[2]*v[1]*(1-math.cos(theta))+v[0]*math.sin(theta), math.cos(theta)+v[2]*v[2]*(1-math.cos(theta))]
        ], dtype=np.float32)
        right_r = R @ right
        up_r    = R @ up_cam
        return v, right_r, up_r

    def draw_camera_frustum(self, center, v, roll, distance=200, fov=60, aspect=16/9):
        """
        Draw frustum rectangle oriented by v and roll.
        """
        fwd, right, up = self.compute_camera_axes(v, roll)
        # half-size
        h2 = distance * math.tan(math.radians(fov)/2)
        w2 = h2 * aspect
        pc = center + fwd * distance
        # corners
        c1 = pc + right*w2 + up*h2
        c2 = pc - right*w2 + up*h2
        c3 = pc - right*w2 - up*h2
        c4 = pc + right*w2 - up*h2
        corners = [c1,c2,c3,c4]
        # draw near plane
        for i in range(4):
            a,b = corners[i], corners[(i+1)%4]
            pts = np.vstack([a,b]).astype(np.float32)
            self.glview.addItem(gl.GLLinePlotItem(pos=pts, color=(0,1,1,1), width=2))
        # connect to camera origin
        for c in corners:
            pts = np.vstack([center,c]).astype(np.float32)
            self.glview.addItem(gl.GLLinePlotItem(pos=pts, color=(1,1,0,1), width=1))
    
    def compute_face_orientation(self, pose):
        """
        Compute face orientation (pitch, yaw, roll) based on midpoint between eyes and ears.
        pose: (n_kp, 3)
        """
        # eye and ear centers
        eye_c = (pose[self.LEFT_EYE] + pose[self.RIGHT_EYE]) * 0.5
        ear_c = (pose[self.LEFT_EAR] + pose[self.RIGHT_EAR]) * 0.5
        # Forward vector from eyes to ears
        fwd_vec = ear_c - eye_c
        fwd = fwd_vec / np.linalg.norm(fwd_vec)
        # Roll vector from left to right ear (tilt)
        ear_vec = pose[self.LEFT_EAR] - pose[self.RIGHT_EAR]
        ev = ear_vec / np.linalg.norm(ear_vec)
        
        # normalize
        v = fwd_vec / np.linalg.norm(fwd_vec)
        # Yaw: rotation around Z (up)
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        # Pitch: rotation around Y (side-to-side)
        pitch = math.degrees(math.atan2(fwd[2], np.linalg.norm(fwd[:2])))
        # Roll: tilt around forward axis (using ear vector)
        roll = math.degrees(math.atan2(ev[2], np.linalg.norm(ev[:2])))
        return eye_c, v, roll

    # def draw_frame(self, idx):
    #     # clear previous non-permanent
    #     for item in list(self.glview.items):
    #         if item not in self.permanent_items:
    #             self.glview.removeItem(item)
                
    #     if self.kp3d is None: return
        
    #     # draw skeleton here...
    #     for i in range(self.n_animal):
    #         pose = self.kp3d[i, idx]
    #         if not np.any(pose): continue
    #         self._draw_pose(pose)
    #         # orientation arrow
    #         eye_c, vdir, roll = self.compute_face_orientation(pose)
    #         self._draw_orientation(eye_c, vdir)
    #         self.draw_camera_frustum(eye_c, -vdir, roll)
    
    def compute_face_axes(self, pose):
        """
        Returns:
          eye_c : numpy array (3,)    — camera center
          fwd   : numpy array (3,)    — normalized forward vector
          up    : numpy array (3,)    — normalized up vector (roll applied)
        """
        # as before, find eye center and ear center
        eye_c = (pose[self.LEFT_EYE] + pose[self.RIGHT_EYE]) * 0.5
        ear_c = (pose[self.LEFT_EAR] + pose[self.RIGHT_EAR]) * 0.5

        # forward vector
        fwd = ear_c - eye_c
        fwd /= np.linalg.norm(fwd)

        # compute a “raw” up vector from ear‐to‐ear, then rotate it onto fwd‐axis
        ear_vec = pose[self.LEFT_EAR] - pose[self.RIGHT_EAR]
        ear_vec /= np.linalg.norm(ear_vec)

        # build orthonormal triad (v, right, up) *before* applying any roll.
        # choose global up to break ties:
        global_up = np.array([0,0,1], dtype=np.float32)
        right = np.cross(global_up, fwd)
        if np.linalg.norm(right) < 1e-6:
            # forward was vertical, pick another UP
            global_up = np.array([0,1,0], dtype=np.float32)
            right = np.cross(global_up, fwd)
        right /= np.linalg.norm(right)

        # preliminary up
        up0 = np.cross(fwd, right)

        # now align up0 with your ear_vec around the fwd axis:
        # compute signed angle between up0 and ear_vec in plane ⟂ fwd
        # θ = atan2( dot( fwd, cross(up0, ear_vec) ), dot(up0, ear_vec) )
        cross_uv = np.cross(up0, ear_vec)
        sinr = np.dot(fwd, cross_uv)
        cosr = np.dot(up0, ear_vec)
        roll_angle = math.atan2(sinr, cosr)
        # rotate up0 around fwd by roll_angle
        c, s = math.cos(roll_angle), math.sin(roll_angle)
        K = np.array([[    0, -fwd[2],  fwd[1]],
                      [ fwd[2],     0, -fwd[0]],
                      [-fwd[1],  fwd[0],     0]], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + s*K + (1-c)*(K@K)
        up = R @ up0

        return eye_c, fwd, right, up
    
    def orientation_from_two(self, v_forward, v_earLR):
        """
        v_forward: 顔の前方向ベクトル
        v_earLR:   耳L→耳R ベクトル（左耳から右耳への向き）

        戻り値: 回転行列 R (3×3)、列ベクトルが [side, up, forward]
        """
        # 1) forward を正規化
        f = v_forward / np.linalg.norm(v_forward)

        # 2) earLR から forward 成分を除去し、side 軸を正規化
        proj = np.dot(v_earLR, f) * f
        s = (v_earLR - proj)
        s /= np.linalg.norm(s)

        # 3) up 軸を外積で得る
        u = np.cross(f, s)
        u /= np.linalg.norm(u)

        # 4) 回転行列を組み立て（列が side, up, forward）
        R = np.stack((s, u, f), axis=1)
        return R

    def draw_camera_frustum_R_cam(self, center, R_cam, distance=200, fov=60, aspect=16/9):
        """
        center : np.array([x,y,z])     — カメラ原点
        R_cam  : np.ndarray shape=(3,3) — 列ベクトルが [right, up, forward]
        distance, fov, aspect はこれまでどおり
        """
        # R_cam の各列を取り出す
        right   = R_cam[:, 0]  # カメラの右方向ベクトル
        up      = R_cam[:, 1]  # カメラの上方向ベクトル
        forward = R_cam[:, 2]  # カメラの前方向ベクトル

        # near 平面の半辺長
        h2 = distance * math.tan(math.radians(fov) / 2)
        w2 = h2 * aspect
        # near 平面の中心
        pc = center + forward * distance

        # ４つのコーナー
        corners = [
            pc + right * w2 + up * h2,
            pc - right * w2 + up * h2,
            pc - right * w2 - up * h2,
            pc + right * w2 - up * h2,
        ]

        # near 平面を描く
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            pts = np.vstack([a, b]).astype(np.float32)
            self.glview.addItem(
                gl.GLLinePlotItem(pos=pts, color=(0, 1, 1, 1), width=2)
            )

        # カメラ原点までの線
        for c in corners:
            pts = np.vstack([center, c]).astype(np.float32)
            self.glview.addItem(
                gl.GLLinePlotItem(pos=pts, color=(1, 1, 0, 1), width=1)
            )

    def draw_frame(self, idx):
        # clear previous non-permanent
        for item in list(self.glview.items):
            if item not in self.permanent_items:
                self.glview.removeItem(item)
        
        if self.kp3d is None:
            return
        
        # for each animal (or just the first one):
        pose = self.kp3d[0, idx]
        if np.any(pose):
            # draw the skeleton in place
            self._draw_pose(pose)
            eye_c, vdir, roll = self.compute_face_orientation(pose)
            ear_vec = pose[self.LEFT_EAR] - pose[self.RIGHT_EAR]
            # compute our camera axes
            eye_c, fwd, right, up = self.compute_face_axes(pose)
            # self._draw_orientation(eye_c, -fwd)
            
            # R_cam = self.orientation_from_two(-fwd,up)
            R_cam = self.orientation_from_two(-fwd,ear_vec)
            self._draw_orientation(eye_c, R_cam[:, 0],color=(1,1,0,1))
            self._draw_orientation(eye_c, R_cam[:, 1],color=(1,0,0,1))
            self._draw_orientation(eye_c, R_cam[:, 2],color=(0,1,0,1))
            # self.draw_camera_frustum_R_cam(eye_c, R_cam)
            
            # # 3) draw the forward/right/up arrows
            # self._draw_axis(eye_c, fwd,   (1,1,0,1))  # forward
            # self._draw_axis(eye_c, -right, (1,0,0,1))  # right
            # self._draw_axis(eye_c, up,    (0,1,0,1))  # upv
            
            # print(f"azimith: {azimuth}, \n elevation: {elevation}, \n roll: {roll}")
            
            
    
    def _draw_axis(self, origin, vec, color, length=50):
        """
        Draws a line from `origin` in direction `vec` of given `length`.
        """
        tip = origin + vec * length
        pts = np.vstack([origin, tip]).astype(np.float32)
        self.glview.addItem(
            gl.GLLinePlotItem(pos=pts, color=color, width=4, antialias=True)
        )


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
            
    def _draw_orientation(self, origin, direction, length=50, color=(1,1,0,1)):
        """
        Draw an arrow from origin in given direction with specified length.
        """
        tip = origin + direction * length
        pts = np.vstack([origin, tip]).astype(np.float32)
        arrow = gl.GLLinePlotItem(pos=pts, color=color, width=4, antialias=True)
        self.glview.addItem(arrow)

    def play_video(self):
        self.timer.start()

    def stop_video(self):
        self.timer.stop()

    def next_frame(self):
        self.goto_frame((self.current_frame+1)%self.n_frame)

    def change_speed(self, v):
        self.timer.setInterval(v)

    def seek_frame(self, idx):
        self.goto_frame(idx)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Pose3DViewer()
    viewer.resize(1200,800)
    viewer.show()
    # load pickle sample
    pf = '../marmo3Dpose/results/3d_v0p8_dark_fix_20/dailylife_cj611_20230226_110000/kp3d_fxdJointLen.pickle'
    viewer.load_pickle(pf)
    sys.exit(app.exec_())
