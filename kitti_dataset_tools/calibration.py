"""Convert existing KITTI P2/P3 calibration to ORB-SLAM3 settings."""
from pathlib import Path
import cv2
import numpy as np

def read_calibration(directory):
    matrices = {}
    for line in (Path(directory) / 'calib.txt').read_text().splitlines():
        key, separator, values = line.partition(':')
        if separator and key in ('P2', 'P3'):
            array = np.fromstring(values, sep=' ')
            if array.size != 12 or not np.isfinite(array).all():
                raise ValueError(f'{key} must contain 12 finite projection-matrix values')
            matrices[key] = array.reshape(3, 4)
    if set(matrices) != {'P2', 'P3'}:
        raise ValueError('calib.txt must contain P2 and P3 for the color cameras')
    left, right = matrices['P2'], matrices['P3']
    for projection in (left, right):
        if projection[0, 0] <= 0 or projection[1, 1] <= 0 or not np.allclose(projection[2, :3], [0, 0, 1]):
            raise ValueError('Invalid rectified camera intrinsics')
    if not np.allclose(left[:3, :3], right[:3, :3], rtol=1e-5, atol=1e-4):
        raise ValueError('This ORB rectified-stereo input requires matching left/right intrinsics')
    # P = K [I | t]. Use both color camera centers, not the gray P0/P1 baseline.
    center_left = -np.linalg.solve(left[:3, :3], left[:, 3])
    center_right = -np.linalg.solve(right[:3, :3], right[:, 3])
    delta = center_right - center_left
    baseline = float(np.linalg.norm(delta))
    if delta[0] <= 0 or baseline <= 0:
        raise ValueError('Expected image_2 on the left and image_3 on the right with positive baseline')
    return left, right, baseline


def write_settings(path, directory, first_left, first_right, timestamps, mode="stereo"):
    left, right, baseline = read_calibration(directory)
    image = cv2.imread(str(first_left), cv2.IMREAD_COLOR)
    other = cv2.imread(str(first_right), cv2.IMREAD_COLOR)
    if image is None or other is None or image.shape != other.shape:
        raise ValueError('Cannot read the first matching stereo pair')
    height, width = image.shape[:2]
    fps = max(1, round(1.0 / float(np.median(np.diff(timestamps))))) if len(timestamps) > 1 else 10
    lines = ['%YAML:1.0', '# KITTI rectified COLOR cameras: calib.txt P2/P3',
             'File.version: "1.0"', 'Camera.type: "Rectified"']
    for index, projection in enumerate((left, right), 1):
        for name, value in zip(('fx', 'fy', 'cx', 'cy'),
                               (projection[0, 0], projection[1, 1], projection[0, 2], projection[1, 2])):
            lines.append(f'Camera{index}.{name}: {float(value):.12f}')
    lines += [f'Camera.width: {width}', f'Camera.height: {height}', f'Camera.fps: {fps}',
              'Camera.RGB: 0', f'Stereo.b: {baseline:.12f}', 'Stereo.ThDepth: 35.0',
              'ORBextractor.nFeatures: 2000', 'ORBextractor.scaleFactor: 1.2',
              'ORBextractor.nLevels: 8', 'ORBextractor.iniThFAST: 20', 'ORBextractor.minThFAST: 7',
              'Viewer.KeyFrameSize: 0.6', 'Viewer.KeyFrameLineWidth: 2.0', 'Viewer.GraphLineWidth: 1.0',
              'Viewer.PointSize: 2.0', 'Viewer.CameraSize: 0.7', 'Viewer.CameraLineWidth: 3.0',
              'Viewer.ViewpointX: 0.0', 'Viewer.ViewpointY: -100.0', 'Viewer.ViewpointZ: -0.1',
              'Viewer.ViewpointF: 2000.0']
    if mode == 'rgbd':
        lines = [line.replace('Camera.type: "Rectified"', 'Camera.type: "PinHole"') for line in lines]
        lines += ['Camera1.k1: 0.0', 'Camera1.k2: 0.0', 'Camera1.p1: 0.0', 'Camera1.p2: 0.0',
                  'Camera1.k3: 0.0', 'RGBD.DepthMapFactor: 1.0']
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n')
    print(f'Wrote {output}: {width}x{height}, {fps} Hz, color baseline {baseline:.6f} m', flush=True)


def main(argv=None):
    import argparse
    import sys
    from .player import read_sequence, read_velodyne_sequence

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset-root', type=Path, help='Dataset root containing sequences/')
    source.add_argument('--sequence-dir', type=Path, help='One sequence directory')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--sequence', help='Single sequence ID, 00..21')
    selection.add_argument('--all-sequences', action='store_true', help='Generate one YAML per existing sequence 00..21')
    parser.add_argument('--mode', choices=('stereo', 'rgbd'), default='stereo')
    parser.add_argument('--settings-output', type=Path, help='Output YAML for a single sequence')
    parser.add_argument('--output-dir', type=Path, help='Batch output directory; default: DATASET_ROOT/config')
    parser.add_argument('--prepare-only', action='store_true', help='Compatibility option; this command never publishes')
    parser.add_argument('--publish-velodyne', action='store_true', help='Also validate point cloud files; no publishing')
    args = parser.parse_args(argv)

    if args.all_sequences:
        if args.dataset_root is None or args.settings_output is not None:
            parser.error('--all-sequences requires --dataset-root; use --output-dir instead of --settings-output')
        root = args.dataset_root.expanduser().resolve()
        output = (args.output_dir or root / 'config').expanduser().resolve()
        directories = [root / 'sequences' / f'{i:02d}' for i in range(22)
                       if (root / 'sequences' / f'{i:02d}').is_dir()]
        if not directories:
            parser.error('No sequence directories 00..21 found under ' + str(root / 'sequences'))
        jobs = [(directory, output / f'kitti_{directory.name}_{"rgbd" if args.mode == "rgbd" else "color"}.yaml') for directory in directories]
    else:
        if args.output_dir is not None or args.settings_output is None:
            parser.error('Single-sequence mode requires --settings-output; --output-dir is for --all-sequences')
        if args.dataset_root is not None:
            if args.sequence is None or not args.sequence.isdecimal() or not 0 <= int(args.sequence) <= 21:
                parser.error('--dataset-root requires --sequence 00..21 or --all-sequences')
            directory = args.dataset_root.expanduser() / 'sequences' / f'{int(args.sequence):02d}'
        else:
            if args.sequence is not None:
                parser.error('Use --sequence with --dataset-root, not --sequence-dir')
            directory = args.sequence_dir.expanduser()
        jobs = [(directory, args.settings_output)]

    failures = []
    for directory, target in jobs:
        try:
            left, right, timestamps = read_sequence(directory)
            if args.publish_velodyne:
                read_velodyne_sequence(directory, left)
            write_settings(target, directory, left[0], right[0], timestamps, mode=args.mode)
        except (ValueError, OSError) as error:
            failures.append(directory.name)
            print(f'ERROR sequence {directory.name}: {error}', file=sys.stderr)
    print(f'Generated {len(jobs) - len(failures)}/{len(jobs)} YAML files.', flush=True)
    if failures:
        print('Failed sequences: ' + ', '.join(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
