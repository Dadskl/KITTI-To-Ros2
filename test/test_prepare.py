"""Exercise installed tools without any SLAM or real KITTI data."""
import subprocess
import sys
import zipfile

import cv2
import numpy as np


def test_prepare_resume_and_calibration_cli(tmp_path):
    def archive(name, entries):
        with zipfile.ZipFile(tmp_path / name, 'w') as z:
            for path, data in entries.items():
                z.writestr(path, data)
    _, image = cv2.imencode('.png', np.zeros((24, 32, 3), dtype=np.uint8))
    archive('data_odometry_color.zip', {
        'dataset/sequences/00/image_2/000000.png': image.tobytes(),
        'dataset/sequences/00/image_3/000000.png': image.tobytes()})
    archive('data_odometry_calib.zip', {
        'dataset/sequences/00/times.txt': '0\n',
        'dataset/sequences/00/calib.txt':
        'P2: 280 0 16 7 0 280 12 0 0 0 1 0\n'
        'P3: 280 0 16 -21 0 280 12 0 0 0 1 0\n'
        'Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n'})
    archive('data_odometry_velodyne.zip', {
        'data_odometry_velodyne/dataset/sequences/00/velodyne/000000.bin': bytes(16)})
    command = [sys.executable, '-m', 'kitti_dataset_tools.prepare', '--archives-dir', str(tmp_path),
               '--output', str(tmp_path / 'out'), '--require-velodyne']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'out/sequences/00/velodyne/000000.bin').read_bytes() == bytes(16)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0 and '0 extracted' in result.stdout
    # Test the installed console entry point, including its prepare-only behavior.
    result = subprocess.run(['ros2', 'run', 'kitti_dataset_tools', 'generate_kitti_settings',
                             '--dataset-root', str(tmp_path / 'out'), '--sequence', '00',
                             '--settings-output', str(tmp_path / 'camera.yaml')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    settings = cv2.FileStorage(str(tmp_path / 'camera.yaml'), cv2.FILE_STORAGE_READ)
    assert abs(settings.getNode('Stereo.b').real() - 0.1) < 1e-9
    settings.release()
    archive('data_odometry_velodyne.zip', {
        'dataset/sequences/00/velodyne/000001.bin': bytes(16)})
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'do not match' in result.stderr


def test_batch_settings(tmp_path):
    from kitti_dataset_tools.calibration import main
    from test_player import make_sequence
    root = tmp_path / 'dataset'
    make_sequence(root / 'sequences/00')
    second = make_sequence(root / 'sequences/02')
    calib = second / 'calib.txt'
    calib.write_text(calib.read_text().replace('320 -21', '320 -49'))
    assert main(['--dataset-root', str(root), '--all-sequences']) == 0
    output = root / 'config'
    assert sorted(p.name for p in output.iterdir()) == ['kitti_00_color.yaml', 'kitti_02_color.yaml']
    for seq, baseline in [('00', 0.1), ('02', 0.2)]:
        settings = cv2.FileStorage(str(output / f'kitti_{seq}_color.yaml'), cv2.FILE_STORAGE_READ)
        assert abs(settings.getNode('Stereo.b').real() - baseline) < 1e-9
        settings.release()
    # One incomplete sequence must not prevent the valid ones from being generated.
    (root / 'sequences/01').mkdir()
    custom = tmp_path / 'custom'
    assert main(['--dataset-root', str(root), '--all-sequences', '--output-dir', str(custom)]) == 1
    assert sorted(p.name for p in custom.iterdir()) == ['kitti_00_color.yaml', 'kitti_02_color.yaml']


def test_batch_arguments(tmp_path):
    import pytest
    from kitti_dataset_tools.calibration import main
    for args in [
        ['--dataset-root', str(tmp_path), '--all-sequences'],
        ['--sequence-dir', str(tmp_path), '--all-sequences'],
        ['--dataset-root', str(tmp_path), '--all-sequences', '--sequence', '00'],
        ['--dataset-root', str(tmp_path), '--all-sequences', '--settings-output', 'all.yaml'],
    ]:
        with pytest.raises(SystemExit):
            main(args)
