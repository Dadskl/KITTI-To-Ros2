"""Validate KITTI color calibration and the actual stereo topic pipeline."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
from kitti_dataset_tools import player


def make_sequence(directory, frames=3):
    for name in ('image_2', 'image_3'):
        (directory / name).mkdir(parents=True)
    rng = np.random.default_rng(19)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8), (3, 3), 0)
    for i in range(frames):
        left = np.roll(texture, i // 3, axis=1)
        right = np.roll(left, -14, axis=1)
        cv2.imwrite(str(directory / 'image_2' / f'{i:06d}.png'), left)
        cv2.imwrite(str(directory / 'image_3' / f'{i:06d}.png'), right)
    (directory / 'times.txt').write_text(''.join(f'{i * 0.1:.6f}\n' for i in range(frames)))
    # Nonzero left projection translation catches the incorrect -P3[0,3]/fx baseline formula.
    (directory / 'calib.txt').write_text(
        'P2: 280 0 320 7 0 280 240 0 0 0 1 0\n'
        'P3: 280 0 320 -21 0 280 240 0 0 0 1 0\n')
    return directory


def test_prepare_color_calibration(tmp_path):
    seq = make_sequence(tmp_path / '00')
    out = tmp_path / 'camera.yaml'
    assert player.main(['--sequence-dir', str(seq), '--settings-output', str(out), '--prepare-only']) == 0
    settings = cv2.FileStorage(str(out), cv2.FILE_STORAGE_READ)
    assert settings.getNode('Camera1.fx').real() == 280.0
    assert settings.getNode('Camera.width').real() == 640
    assert settings.getNode('Camera.RGB').real() == 0
    assert settings.getNode('Stereo.b').real() == pytest.approx(0.1)
    settings.release()


def test_reject_missing_pair(tmp_path):
    seq = make_sequence(tmp_path / '00')
    (seq / 'image_3' / '000001.png').unlink()
    with pytest.raises(ValueError, match='matching'):
        player.read_sequence(seq)


def test_select_sequence(tmp_path):
    make_sequence(tmp_path / 'sequences' / '03')
    out = tmp_path / 'selected.yaml'
    assert player.main(['--dataset-root', str(tmp_path), '--sequence', '3',
                        '--settings-output', str(out), '--prepare-only']) == 0
    assert out.is_file()
    with pytest.raises(SystemExit):
        player.main(['--dataset-root', str(tmp_path), '--sequence', '../03'])


def test_reject_repeated_timestamps(tmp_path):
    seq = make_sequence(tmp_path / '00')
    (seq / 'times.txt').write_text('0\n0\n0.2\n')
    with pytest.raises(ValueError, match='strictly increasing'):
        player.read_sequence(seq)


def test_velodyne_validation_and_message(tmp_path):
    from builtin_interfaces.msg import Time
    seq = make_sequence(tmp_path / '00')
    images, _, _ = player.read_sequence(seq)
    with pytest.raises(ValueError, match='matching'):
        player.read_velodyne_sequence(seq, images)
    (seq / 'velodyne').mkdir()
    points = np.array([[1, -2, 3, 0.5], [4, 5, 6, 0.25]], dtype='<f4')
    for image in images:
        (seq / 'velodyne' / (image.stem + '.bin')).write_bytes(points.tobytes())
    scans = player.read_velodyne_sequence(seq, images)
    stamp = Time(sec=12, nanosec=345)
    msg = player.read_velodyne_message(scans[0], stamp, 'kitti_velodyne')
    assert msg.header.stamp == stamp and msg.header.frame_id == 'kitti_velodyne'
    assert (msg.width, msg.height, msg.point_step, msg.row_step) == (2, 1, 16, 32)
    assert [(f.name, f.offset, f.datatype, f.count) for f in msg.fields] == [
        (name, offset, 7, 1) for name, offset in zip(('x', 'y', 'z', 'intensity'), (0, 4, 8, 12))]
    assert not msg.is_bigendian and msg.is_dense
    np.testing.assert_array_equal(np.frombuffer(bytes(msg.data), dtype='<f4').reshape(-1, 4), points)
    points[0, 0] = np.nan
    scans[0].write_bytes(points.tobytes())
    assert not player.read_velodyne_message(scans[0], stamp, 'kitti_velodyne').is_dense
    scans[0].write_bytes(b'bad')
    with pytest.raises(ValueError, match='Invalid Velodyne'):
        player.read_velodyne_sequence(seq, images)
    with pytest.raises(ValueError, match='Invalid Velodyne'):
        player.read_velodyne_message(scans[0], stamp, 'kitti_velodyne')


@pytest.mark.parametrize('mode,reliability,clock', [('stereo', 'reliable', True), ('mono', 'best_effort', False), ('rgbd', 'reliable', True)])
def test_velodyne_topics(tmp_path, mode, reliability, clock):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, PointCloud2
    seq = make_sequence(tmp_path / '00', frames=4)
    # Small images keep this transport test independent of UDP fragmentation limits.
    for image in seq.glob('image_*/*.png'):
        cv2.imwrite(str(image), np.zeros((24, 32, 3), dtype=np.uint8))
    with (seq / 'calib.txt').open('a') as f:
        f.write('Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
    (seq / 'velodyne').mkdir()
    for i in range(4):
        np.array([[i, 2, 3, 0.5]], dtype='<f4').tofile(seq / 'velodyne' / f'{i:06d}.bin')
    domain = 40 + os.getpid() % 150
    context = rclpy.context.Context()
    rclpy.init(context=context, domain_id=domain)
    node = rclpy.create_node('velodyne_smoke', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    received = {'left': [], 'right': [], 'cloud': [], 'depth': []}
    subscriptions = [node.create_subscription(kind, topic, lambda msg, k=key: received[k].append(msg),
                                              qos_profile_sensor_data)
                     for key, kind, topic in [('left', Image, '/kitti/left/image_raw'),
                                              ('right', Image, '/kitti/right/image_raw'),
                                              ('cloud', PointCloud2, '/test/velodyne'), ('depth', Image, '/kitti/depth/image_raw')]]
    command = [sys.executable, '-m', 'kitti_dataset_tools.player', '--sequence-dir', str(seq),
               '--publish-velodyne', '--velodyne-topic', '/test/velodyne', '--velodyne-frame', 'test_lidar',
               '--start-frame', '1', '--max-frames', '2', '--speed', '0.5', '--mode', mode,
               '--reliability', reliability, '--wait-for-subscribers', '10']
    if mode == 'rgbd':
        command.remove('--publish-velodyne')  # RGB-D must publish LiDAR automatically.
        command += ['--depth-source', 'lidar']
    if clock:
        command.append('--publish-clock')
    process = None
    try:
        with (tmp_path / 'player.log').open('w+') as log:
            process = subprocess.Popen(command, env=dict(os.environ, ROS_DOMAIN_ID=str(domain)), stdout=log, stderr=log)
            deadline = time.monotonic() + 15
            while process.poll() is None and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
            assert process.poll() == 0, (tmp_path / 'player.log').read_text()
            until = time.monotonic() + 0.5
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=0.05)
        assert len(received['cloud']) == len(received['left']) == 2
        assert len(received['right']) == (2 if mode != 'mono' else 0)
        if mode == 'rgbd':
            assert len(received['depth']) == 2
            for depth, color in zip(received['depth'], received['left']):
                assert depth.header == color.header
                assert depth.encoding == '32FC1' and depth.step == depth.width * 4
                assert (depth.height, depth.width) == (color.height, color.width)
        for i, (cloud, left) in enumerate(zip(received['cloud'], received['left']), 1):
            assert cloud.header.stamp == left.header.stamp
            assert cloud.header.frame_id == 'test_lidar'
            assert np.frombuffer(bytes(cloud.data), dtype='<f4')[0] == i
            if mode != 'mono':
                assert cloud.header.stamp == received['right'][i - 1].header.stamp
            if clock:
                assert cloud.header.stamp.sec == 0
                assert cloud.header.stamp.nanosec == i * 100000000
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
