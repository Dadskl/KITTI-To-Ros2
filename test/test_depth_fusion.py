import cv2
import numpy as np
import pytest
from kitti_dataset_tools.depth_fusion import DepthFusion, FusionConfig
from kitti_dataset_tools.calibration import write_settings
from test_player import make_sequence


def fixture(tmp_path, source='fused'):
    seq = make_sequence(tmp_path / '00')
    with (seq / 'calib.txt').open('a') as f:
        f.write('Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
    return seq, DepthFusion(seq, FusionConfig(num_disparities=32), source)


def test_projection_zbuffer(tmp_path):
    _, fusion = fixture(tmp_path)
    points = np.array([[0, 0, 10, 1], [0, 0, 20, 1], [0, 0, -1, 1], [np.nan, 0, 1, 1]])
    depth = fusion.lidar_depth(points, (480, 640))
    assert depth[240, 320] == 10
    assert np.count_nonzero(depth) == 1


def test_stereo_metric_depth(tmp_path):
    _, fusion = fixture(tmp_path, 'stereo')
    rng = np.random.default_rng(42)
    left = rng.integers(0, 256, (100, 320, 3), dtype=np.uint8)
    right = np.roll(left, -14, axis=1)
    result = fusion.compute(left, right)['depth']
    good = result[:, 50:280]
    assert np.count_nonzero(good) > good.size * 0.7
    assert np.median(good[good > 0]) == pytest.approx(2, abs=0.03)


def test_occlusion_and_holes(tmp_path):
    _, fusion = fixture(tmp_path)
    fusion.stereo_depth = lambda a, b: np.array([[2, 0, 10]], np.float32)
    fusion.lidar_depth = lambda p, s: np.array([[10, 5, 9]], np.float32)
    image = np.zeros((1, 3, 3), np.uint8)
    np.testing.assert_equal(fusion.compute(image, image, [])['depth'], [[2, 5, 9]])


def test_rgbd_config_and_invalid_params(tmp_path):
    seq, _ = fixture(tmp_path)
    output = tmp_path / 'rgbd.yaml'
    write_settings(output, seq, seq/'image_2/000000.png', seq/'image_3/000000.png', np.array([0, .1]), mode='rgbd')
    f = cv2.FileStorage(str(output), cv2.FILE_STORAGE_READ)
    assert f.getNode('Camera.type').string() == 'PinHole'
    assert f.getNode('RGBD.DepthMapFactor').real() == 1
    f.release()
    with pytest.raises(ValueError):
        FusionConfig(num_disparities=17).validate()
