"""Probe the top RealSense: model, supported streams, and depth quality at the table (read-only, nothing moves).
    python classical\\probe_depth.py      -> prints info, saves test_images\\depth_*.png and depth_frame.npz
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = Path(__file__).parent
SERIAL = os.environ.get("CAM_TOP", "218622276310")

ctx = rs.context()
dev = None
for d in ctx.query_devices():
    print("device:", d.get_info(rs.camera_info.name), "| serial", d.get_info(rs.camera_info.serial_number),
          "| firmware", d.get_info(rs.camera_info.firmware_version))
    if d.get_info(rs.camera_info.serial_number) == SERIAL:
        dev = d
if dev is None:
    sys.exit(f"top camera {SERIAL} not found")
for s in dev.query_sensors():
    name = s.get_info(rs.camera_info.name)
    profs = sorted({(p.stream_name(), p.as_video_stream_profile().width(), p.as_video_stream_profile().height(), p.fps())
                    for p in s.get_stream_profiles() if p.is_video_stream_profile()})
    best = {}
    for sn, w, h, f in profs:
        if f == 30:
            best.setdefault(sn, set()).add((w, h))
    print(f"  sensor {name}: 30 fps sizes", {k: sorted(v)[-4:] for k, v in best.items()})

pipe, cfg = rs.pipeline(), rs.config()
cfg.enable_device(SERIAL)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = pipe.start(cfg)
scale = prof.get_device().first_depth_sensor().get_depth_scale()
align = rs.align(rs.stream.color)
spat, temp = rs.spatial_filter(), rs.temporal_filter()
for _ in range(40):
    fr = align.process(pipe.wait_for_frames())
depths = []
for _ in range(15):
    fr = align.process(pipe.wait_for_frames())
    d = fr.get_depth_frame()
    d = temp.process(spat.process(d)).as_depth_frame()
    depths.append(np.asanyarray(d.get_data()).astype(np.float32) * scale)
color = np.asanyarray(fr.get_color_frame().get_data()).copy()
intr = fr.get_color_frame().profile.as_video_stream_profile().intrinsics
pipe.stop()
D = np.median(np.stack(depths), 0)
valid = D > 0
print(f"depth scale {scale} m | valid pixels {valid.mean() * 100:.0f}% | intrinsics fx {intr.fx:.1f} fy {intr.fy:.1f} "
      f"cx {intr.ppx:.1f} cy {intr.ppy:.1f}")
table = D[300:380, 250:450]
t = table[table > 0]
print(f"table patch: median distance {np.median(t) * 100:.1f} cm, noise (std) {t.std() * 1000:.1f} mm, valid {np.mean(table > 0) * 100:.0f}%")
np.savez_compressed(HERE / "test_images" / "depth_frame.npz", depth=D, color=color,
                    intr=np.array([intr.fx, intr.fy, intr.ppx, intr.ppy]))
cv2.imwrite(str(HERE / "test_images" / "depth_color.png"), color)
vis = np.clip((np.median(t) - D) * 1000 * 5, 0, 255).astype(np.uint8)   # height above the table, 1 grey level = 0.2 mm
vis[~valid] = 0
cv2.imwrite(str(HERE / "test_images" / "depth_height.png"), cv2.applyColorMap(vis, cv2.COLORMAP_JET))
print("saved test_images/depth_color.png, depth_height.png, depth_frame.npz")
