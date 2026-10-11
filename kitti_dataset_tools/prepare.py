#!/usr/bin/env python3
"""Extract KITTI odometry ZIPs into a unified dataset. Includes optional Velodyne point clouds. Standard library only."""
import argparse
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import zipfile
import zlib


class Progress:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.start = self.last = time.monotonic()

    def update(self, count=0, force=False):
        self.done += count
        now = time.monotonic()
        if not force and now - self.last < 0.15:
            return
        self.last = now
        fraction = self.done / self.total if self.total else 1
        rate = self.done / max(now - self.start, 0.001)
        eta = (self.total - self.done) / rate if rate else 0
        filled = min(30, int(30 * fraction))
        print('\r[{}{}] {:6.2f}%  {:.2f}/{:.2f} GiB  {:.1f} MiB/s  ETA {:.0f}s'.format(
            '#' * filled, '-' * (30 - filled), fraction * 100,
            self.done / 2**30, self.total / 2**30, rate / 2**20, eta),
            end='', flush=True)


def safe_target(root, relative):
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith('/') or '\\' in relative or any(p in ('..', '.') or ':' in p for p in parts):
        raise ValueError('Unsafe archive path: ' + relative)
    target = root.joinpath(*parts)
    if root != target.resolve() and root not in target.resolve().parents:
        raise ValueError('Path escapes output directory: ' + relative)
    for component in [target] + list(target.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError('Symlink in output path: ' + str(component))
    return target


def matches(path, info):
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    crc = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xffffffff == info.CRC


def validate(root, sequences, with_velodyne=False):
    lines = ['KITTI extraction validation', 'Extracted files checked by ZIP CRC; pixel decoding not checked.']
    for seq in sorted(sequences):
        directory = root / 'sequences' / seq
        left = sorted(p.name for p in (directory / 'image_2').glob('*.png'))
        right = sorted(p.name for p in (directory / 'image_3').glob('*.png'))
        if not left or left != right or left != ['{:06d}.png'.format(i) for i in range(len(left))]:
            raise ValueError('Missing/noncontiguous/mismatched stereo frames in sequence ' + seq)
        times = [float(t) for t in (directory / 'times.txt').read_text().split()]
        if len(times) != len(left) or any(not math.isfinite(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Invalid timestamps for sequence ' + seq)
        calibration = {}
        for line in (directory / 'calib.txt').read_text().splitlines():
            key, separator, values = line.partition(':')
            if separator:
                calibration[key] = values.split()
        for key in (('P2', 'P3', 'Tr') if with_velodyne else ('P2', 'P3')):
            values = calibration.get(key, [])
            if len(values) != 12 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError('Invalid {} in sequence {}'.format(key, seq))
        if with_velodyne:
            scans = sorted((directory / 'velodyne').glob('*.bin'))
            if [p.stem for p in scans] != [Path(n).stem for n in left]:
                raise ValueError('Missing/mismatched Velodyne frames in sequence ' + seq)
            for scan in scans:
                # KITTI stores each point as four float32 values: x, y, z, reflectance.
                if scan.stat().st_size == 0 or scan.stat().st_size % 16:
                    raise ValueError('Invalid Velodyne binary size: ' + str(scan))
            lines.append('{}: {} Velodyne scans, frame alignment/record sizes/Tr OK'.format(seq, len(scans)))
        pose = root / 'poses' / (seq + '.txt')
        if pose.exists():
            pose_lines = [line.split() for line in pose.read_text().splitlines() if line.strip()]
            if len(pose_lines) != len(left) or any(len(row) != 12 or not all(math.isfinite(float(v)) for v in row) for row in pose_lines):
                raise ValueError('Invalid ground truth poses for sequence ' + seq)
        lines.append('{}: {} stereo pairs, timestamps/calibration OK, ground truth {}'.format(
            seq, len(left), 'present' if pose.exists() else 'absent (normal for test sequences)'))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives-dir', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('kitti_dataset'), help='Dataset root; sequences/ and poses/ go directly here')
    parser.add_argument('--sequences', nargs='+', help='Optional sequence IDs, e.g. 00 01; default: all')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing files whose size/CRC differs')
    parser.add_argument('--require-velodyne', action='store_true', help='Fail if data_odometry_velodyne.zip is absent; otherwise include it automatically when present')
    args = parser.parse_args()
    opened = []
    try:
        chosen = None
        if args.sequences:
            if any(not s.isdigit() or not 0 <= int(s) <= 21 for s in args.sequences):
                raise ValueError('Sequence IDs must be between 00 and 21')
            chosen = {str(int(s)).zfill(2) for s in args.sequences}
        root = args.output.expanduser().resolve()
        source = args.archives_dir.expanduser().resolve()
        plan = {}
        sequences = set()
        # Later calibration entries intentionally supersede copies in the color ZIP.
        names = ['data_odometry_color.zip', 'data_odometry_calib.zip',
                 'data_odometry_poses.zip', 'devkit_odometry.zip', 'data_odometry_velodyne.zip']
        with_velodyne = (source / 'data_odometry_velodyne.zip').is_file()
        if args.require_velodyne and not with_velodyne:
            raise ValueError('Required archive missing: ' + str(source / 'data_odometry_velodyne.zip'))
        for name in names:
            path = source / name
            if not path.is_file():
                if name in names[:2]:
                    raise ValueError('Required archive missing: ' + str(path))
                print('Optional archive absent: ' + name)
                continue
            archive = zipfile.ZipFile(path)
            opened.append(archive)
            seen = set()
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError('Archive symlink not supported: ' + info.filename)
                safe_target(root, info.filename)
                parts = PurePosixPath(info.filename).parts
                if parts[0] == Path(name).stem:
                    parts = parts[1:]
                if parts and parts[0] == 'dataset':
                    parts = parts[1:]
                if not parts:
                    continue
                if name == 'devkit_odometry.zip':
                    relative = '/'.join(parts) if parts[0] == 'devkit' else 'devkit/' + '/'.join(parts)
                else:
                    if parts[0] == 'sequences' and len(parts) >= 3:
                        seq = parts[1]
                        if chosen and seq not in chosen:
                            continue
                        if name == names[0] and 'image_2' in parts:
                            sequences.add(seq)
                    elif parts[0] == 'poses' and len(parts) == 2:
                        if chosen and PurePosixPath(parts[1]).stem not in chosen:
                            continue
                    else:
                        raise ValueError('Unexpected dataset layout: ' + info.filename)
                    relative = '/'.join(parts)
                if name == 'data_odometry_velodyne.zip':
                    # Only point clouds belong to this archive; metadata comes from calib ZIP.
                    if len(parts) != 4 or parts[0] != 'sequences' or parts[2] != 'velodyne' or not parts[3].endswith('.bin'):
                        raise ValueError('Unexpected Velodyne layout: ' + info.filename)
                    if info.file_size == 0 or info.file_size % 16:
                        raise ValueError('Invalid Velodyne binary size: ' + info.filename)
                if relative in seen:
                    raise ValueError('Duplicate entry in archive: ' + info.filename)
                seen.add(relative)
                if relative in plan and name != names[1]:
                    raise ValueError('Unexpected overlapping entry: ' + relative)
                plan[relative] = (archive, info)
        if not sequences or (chosen and sequences != chosen):
            raise ValueError('Requested color sequences not present in archives')
        if with_velodyne:
            # Fail before extracting a large archive if modalities do not align.
            image_frames = {}
            lidar_frames = {}
            for relative in plan:
                parts = PurePosixPath(relative).parts
                if len(parts) == 4 and parts[0] == 'sequences':
                    if parts[2] == 'image_2' and parts[3].endswith('.png'):
                        image_frames.setdefault(parts[1], set()).add(Path(parts[3]).stem)
                    elif parts[2] == 'velodyne':
                        lidar_frames.setdefault(parts[1], set()).add(Path(parts[3]).stem)
            if image_frames != lidar_frames:
                raise ValueError('Velodyne and color frame IDs/sequences do not match in archives')
        root.mkdir(parents=True, exist_ok=True)
        total = sum(info.file_size for _, info in plan.values())
        needed = sum(info.file_size for relative, (_, info) in plan.items()
                     if not safe_target(root, relative).is_file())
        free = shutil.disk_usage(root).free
        if free < needed + max(info.file_size for _, info in plan.values()):
            raise ValueError('Insufficient free disk space: need at least {:.2f} GiB plus replacement space'.format(needed / 2**30))
        print('Output: {}\nSequences: {}\nFiles: {}; logical size: {:.2f} GiB'.format(
            root, ', '.join(sorted(sequences)), len(plan), total / 2**30))
        progress = Progress(total)
        written = skipped = 0
        for relative, (archive, info) in plan.items():
            target = safe_target(root, relative)
            if target.exists():
                if matches(target, info):
                    skipped += 1
                    progress.update(info.file_size)
                    continue
                if not args.overwrite:
                    raise ValueError('Existing file differs: {}. Use a new output directory or --overwrite.'.format(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.kitti-', suffix='.part', delete=False) as out:
                    temporary = Path(out.name)
                    with archive.open(info) as inp:
                        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
                            out.write(chunk)
                            progress.update(len(chunk))
                # ZIP reader checks CRC before this atomic replacement.
                os.replace(str(temporary), str(target))
                written += 1
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
        progress.update(force=True)
        print('\nValidating stereo pairs, timestamps, calibration and available point clouds...')
        report = validate(root, sequences, with_velodyne)
        (root / 'preparation_report.txt').write_text(report, encoding='utf-8')
        print(report)
        print('Complete: {} extracted, {} verified and skipped.'.format(written, skipped))
        return 0
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print('\nERROR: {}'.format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nInterrupted. Run the same command again to resume.', file=sys.stderr)
        return 130
    finally:
        for archive in opened:
            archive.close()


if __name__ == '__main__':
    raise SystemExit(main())
