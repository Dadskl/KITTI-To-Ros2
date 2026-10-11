"""ROS-independent, conservative stereo/Velodyne depth fusion in left camera metres.

No learned completion, spatial LiDAR dilation, or motion compensation is performed.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import read_calibration


@dataclass
class FusionConfig:
    min_depth: float = 0.5
    max_depth: float = 80.0
    num_disparities: int = 128
    block_size: int = 5
    lr_threshold: float = 1.0
    occlusion_tolerance: float = 0.15

    def validate(self):
        values = [self.min_depth, self.max_depth, self.lr_threshold, self.occlusion_tolerance]
        if not np.isfinite(values).all() or not 0 < self.min_depth < self.max_depth:
            raise ValueError('Depth limits must be finite with 0 < min < max')
        if self.num_disparities < 16 or self.num_disparities % 16:
            raise ValueError('num_disparities must be a positive multiple of 16')
        if self.block_size < 3 or self.block_size > 21 or self.block_size % 2 == 0:
            raise ValueError('block_size must be odd, between 3 and 21')
        if self.lr_threshold <= 0 or self.occlusion_tolerance < 0:
            raise ValueError('Invalid consistency or occlusion threshold')


class DepthFusion:
    def __init__(self, directory, config=None, source='fused'):
        self.config = config or FusionConfig()
        self.config.validate()
        if source not in ('fused', 'stereo', 'lidar'):
            raise ValueError('Unknown depth source')
        self.source = source
        self.left, self.right, _ = read_calibration(directory)
        self.t_left = np.linalg.solve(self.left[:, :3], self.left[:, 3])
        self.t_right = np.linalg.solve(self.right[:, :3], self.right[:, 3])
        self.transform = None
        if source != 'stereo':
            for line in (Path(directory) / 'calib.txt').read_text().splitlines():
                key, _, value = line.partition(':')
                if key == 'Tr':
                    values = np.fromstring(value, sep=' ')
                    if values.size != 12 or not np.isfinite(values).all():
                        raise ValueError('Tr must contain 12 finite values')
                    self.transform = np.eye(4)
                    self.transform[:3] = values.reshape(3, 4)
            if self.transform is None:
                raise ValueError('LiDAR depth requires Tr in calib.txt')
        c = self.config
        def matcher(min_disparity):
            return cv2.StereoSGBM_create(
                minDisparity=min_disparity, numDisparities=c.num_disparities,
                blockSize=c.block_size, P1=8*c.block_size**2, P2=32*c.block_size**2,
                disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=50,
                speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.matcher = matcher(0)
        self.reverse_matcher = matcher(-c.num_disparities + 1)

    def stereo_depth(self, left, right):
        c = self.config
        if left.shape[1] <= c.num_disparities + c.block_size:
            raise ValueError('Image width must exceed disparity search range plus block size')
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16
        reverse = self.reverse_matcher.compute(gray_r, gray_l).astype(np.float32) / 16
        y, x = np.indices(disparity.shape)
        xr = np.rint(x - disparity).astype(np.int32)
        sampled = reverse[y, np.clip(xr, 0, left.shape[1]-1)]
        valid = (disparity > 0) & (xr >= 0) & (xr < left.shape[1])
        valid &= (sampled > -c.num_disparities) & (sampled <= 0)
        valid &= np.abs(disparity + sampled) <= c.lr_threshold
        # P_i = K_i [I | t_i]. Include projection translations, including t_z.
        xl = (x-self.left[0, 2])/self.left[0, 0]
        rr = (x-disparity-self.right[0, 2])/self.right[0, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = ((self.t_left[0]-self.t_right[0] - xl*self.t_left[2]
                  + rr*self.t_right[2]) / (xl-rr) + self.t_left[2])
        valid &= np.isfinite(z) & (z >= c.min_depth) & (z <= c.max_depth)
        return np.where(valid, z, 0).astype(np.float32)

    def lidar_depth(self, points, shape):
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            raise ValueError('LiDAR input must be N x 3 or N x 4')
        xyz = points[:, :3]
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        reference = np.column_stack((xyz, np.ones(len(xyz)))) @ self.transform.T
        projected = reference @ self.left.T
        z = reference[:, 2] + self.t_left[2]
        valid = (z >= self.config.min_depth) & (z <= self.config.max_depth)
        projected, z = projected[valid], z[valid]
        uv = projected[:, :2] / projected[:, 2:3]
        height, width = shape
        valid = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        valid &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
        uv, z = uv[valid].astype(np.int32), z[valid]
        depth = np.full(height*width, np.inf, dtype=np.float32)
        # Nearest camera Z wins when multiple returns land on one pixel.
        np.minimum.at(depth, uv[:, 1]*width + uv[:, 0], z)
        depth[~np.isfinite(depth)] = 0
        return depth.reshape(shape)

    def compute(self, left, right, points=None):
        if left is None or right is None or left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise ValueError('Expected matching BGR stereo images')
        shape = left.shape[:2]
        stereo = self.stereo_depth(left, right) if self.source != 'lidar' else np.zeros(shape, np.float32)
        lidar = self.lidar_depth(points, shape) if self.source != 'stereo' else np.zeros(shape, np.float32)
        fused = stereo.copy()
        use = lidar > 0
        # Do not paint a distant LiDAR return through a nearer stereo foreground.
        use &= (stereo == 0) | (lidar <= stereo * (1+self.config.occlusion_tolerance))
        fused[use] = lidar[use]
        return {'depth': fused, 'stereo': stereo, 'lidar': lidar}
