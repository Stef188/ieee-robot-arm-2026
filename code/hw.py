"""Hardware for the classical sorter: both follower arms (LeRobot SOFollower, same calibration files and motor
settings as the competition runs) and the top RealSense camera. Run through config.bat (ports, camera serial).

Motion is always slow and smooth: move_to() interpolates in joint space at <= MAX_SPEED deg/s and waits until the
arm is there. Joint targets are clipped to the range seen during calibration (+ a margin), so a bad target can
never swing an arm somewhere it has never been.
"""
import os
import time

import numpy as np

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MAX_SPEED = float(os.environ.get("SORT_SPEED") or 45.0)     # deg/s (gripper too)
RATE = 50.0                                                 # commands per second
GRIP_OPEN = float(os.environ.get("SORT_GRIP_OPEN") or 65.0)   # 25 Nov: was 40 - too narrow for the large cube
GRIP_CLOSED = 1.5
GRIP_SPEED = float(os.environ.get("SORT_GRIP_SPEED") or 0)   # 26 Nov: >0 = the gripper opens/closes at this speed of
#   its own (fast) instead of being held to the joints' deg/s - closing used to take ~1.6 s


class Arm:
    def __init__(self, side):
        self.side = side
        port = os.environ["LEFT_FOLLOWER" if side == "left" else "RIGHT_FOLLOWER"]
        self.robot = SOFollower(SOFollowerRobotConfig(id=f"bimanual_follower_{side}", port=port, use_degrees=True,
                                                 num_read_retries=10, max_relative_target=None))
        self.limits = None          # {joint: (lo, hi)} allowed targets

    def connect(self):
        self.robot.connect(calibrate=False)

    def disconnect(self):
        try:
            self.robot.disconnect()
        except Exception as e:
            print(f"{self.side}: disconnect: {e!r}")

    def read(self):
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=float)

    def torque(self, on):
        if on:
            self.robot.bus.enable_torque()
        else:
            self.robot.bus.disable_torque()

    def send(self, q):
        q = np.asarray(q, float).copy()
        if self.limits:
            for i, j in enumerate(JOINTS):
                if j in self.limits:
                    lo, hi = self.limits[j]
                    q[i] = min(max(q[i], lo), hi)
        self.robot.send_action({f"{j}.pos": float(q[i]) for i, j in enumerate(JOINTS)})
        return q

    def move_to(self, target, speed=MAX_SPEED, settle=0.25, tol=3.0, timeout=8.0, stall=False):
        """Glide to `target` (6 values; None entries keep the current value).
        stall=True (26 Nov pm): also stop waiting as soon as the arm has STOPPED moving (blocked by the table or a bin
        rim, or sagging) - instead of pushing against it until the timeout."""
        start = self.read()
        tgt = np.array([start[i] if t is None else t for i, t in enumerate(target)], float)
        if GRIP_SPEED > 0:
            dist_t = max(np.max(np.abs(tgt[:5] - start[:5])) / speed, abs(tgt[5] - start[5]) / GRIP_SPEED)
            steps = max(int(dist_t * RATE), 1)
        else:
            dist = np.max(np.abs(tgt - start))
            steps = max(int(dist / speed * RATE), 1)
        for s in range(1, steps + 1):
            self.send(start + (tgt - start) * s / steps)
            time.sleep(1 / RATE)
        t0 = time.time()
        prev, still = None, 0
        while time.time() - t0 < timeout:
            q = self.read()[:5]
            err = np.max(np.abs(q - tgt[:5]))
            if err <= tol and time.time() - t0 >= settle:
                return err
            if stall:
                still = still + 1 if prev is not None and np.max(np.abs(q - prev)) < 0.25 else 0
                if still >= 3 and time.time() - t0 >= settle:
                    return err
            prev = q
            time.sleep(0.05)
        return err


class TopCamera:
    def __init__(self):
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        self.cam = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=os.environ.get("CAM_TOP", "218622276310"),
                                                         width=640, height=480, fps=30))

    def connect(self):
        self.cam.connect()
        for _ in range(30):         # let exposure settle
            self.cam.read()

    def grab(self):
        import cv2
        for _ in range(3):          # flush old frames
            self.cam.read()
        return cv2.cvtColor(np.asarray(self.cam.read()), cv2.COLOR_RGB2BGR)

    def disconnect(self):
        try:
            self.cam.disconnect()
        except Exception:
            pass


class WristCamera(TopCamera):
    """The camera on one gripper, opened exactly like the competition runs (config.bat: LEFT/RIGHT_WRIST_TYPE,
    CAM_LEFT_WRIST / CAM_RIGHT_WRIST). Left = RealSense 424x240, right = Logitech webcam 640x360 (MJPG, DirectShow)."""

    def __init__(self, side):
        S = side.upper()
        kind = os.environ.get(f"{S}_WRIST_TYPE") or os.environ.get("WRIST_TYPE", "intelrealsense")
        spec = os.environ.get(f"CAM_{S}_WRIST", "")
        fps = int(os.environ.get("FPS", 30))
        if kind == "opencv":
            from lerobot.cameras.configs import Cv2Backends
            from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
            self.cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=int(spec), width=640, height=360, fps=fps,
                                                       fourcc="MJPG", backend=Cv2Backends(700)))
        else:
            from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
            self.cam = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name=spec, width=424, height=240, fps=fps))
