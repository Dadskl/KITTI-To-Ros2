#!/usr/bin/env python3
"""Inspect KITTI ZIP archives without extracting them; Python standard library only."""

import argparse
import collections
import math
from pathlib import Path, PurePosixPath
import statistics
import struct
import zipfile


def inspect_archive(path, emit):
    emit('\n' + '=' * 72)
    emit('ARCHIVE: ' + path.name)
    emit('Archive bytes: ' + str(path.stat().st_size))
    with zipfile.ZipFile(path) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        emit('Files: {}; uncompressed bytes: {}'.format(
            len(files), sum(item.file_size for item in files)))
        extensions = collections.Counter(PurePosixPath(i.filename).suffix.lower() or '(none)' for i in files)
        emit('Extensions: ' + str(dict(extensions)))
        folders = collections.Counter(str(PurePosixPath(i.filename).parent) for i in files)
        emit('\nDIRECTORIES (direct file counts):')
        for folder, count in sorted(folders.items()):
            emit('  {} : {}'.format(folder, count))
        duplicates = [name for name, n in collections.Counter(i.filename for i in files).items() if n > 1]
        emit('Duplicate ZIP entry names: {}'.format(len(duplicates)))
        for name in duplicates[:10]:
            emit('  ' + name)

        images = collections.defaultdict(list)
        text_files = []
        for item in files:
            name = PurePosixPath(item.filename)
            if name.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                images[str(name.parent)].append(item)
            if name.suffix.lower() == '.txt':
                text_files.append(item)

        emit('\nIMAGE GROUPS:')
        for folder, entries in sorted(images.items()):
            entries.sort(key=lambda item: item.filename)
            stems = [PurePosixPath(i.filename).stem for i in entries]
            emit('  {}: {} images; first={}, last={}'.format(
                folder, len(entries), PurePosixPath(entries[0].filename).name,
                PurePosixPath(entries[-1].filename).name))
            if all(s.isdecimal() for s in stems):
                numbers = sorted(int(s) for s in stems)
                missing = numbers[-1] - numbers[0] + 1 - len(set(numbers))
                emit('    Frame range {}..{}; missing indices={}; starts at zero={}'.format(
                    numbers[0], numbers[-1], missing, numbers[0] == 0))
            # Only inspect one header per directory, never decompress the image collection.
            try:
                with archive.open(entries[0]) as stream:
                    header = stream.read(32)
                if header[:8] == b'\x89PNG\r\n\x1a\n' and len(header) >= 29:
                    width, height, bits, color = struct.unpack('>IIBB', header[16:26])
                    kinds = {0: 'grayscale', 2: 'RGB', 3: 'palette', 4: 'gray+alpha', 6: 'RGBA'}
                    emit('    Sample PNG: {}x{}, {} bit, {} (color type {})'.format(
                        width, height, bits, kinds.get(color, 'unknown'), color))
                else:
                    emit('    Sample header: {}; dimensions not parsed for this format'.format(header[:16].hex()))
            except Exception as exc:
                emit('    ERROR reading sample: {}: {}'.format(type(exc).__name__, exc))

        emit('\nSTEREO PAIRS (within this archive):')
        for folder, left in sorted(images.items()):
            if PurePosixPath(folder).name != 'image_2':
                continue
            right_folder = str(PurePosixPath(folder).parent / 'image_3')
            if right_folder not in images:
                emit('  {}: image_3 not found in this archive'.format(folder))
                continue
            left_names = {PurePosixPath(i.filename).name for i in left}
            right_names = {PurePosixPath(i.filename).name for i in images[right_folder]}
            emit('  {}: paired={}, left-only={}, right-only={}'.format(
                PurePosixPath(folder).parent, len(left_names & right_names),
                len(left_names - right_names), len(right_names - left_names)))

        emit('\nTEXT FILES (bounded previews):')
        for item in sorted(text_files, key=lambda i: i.filename):
            emit('  {} ({} bytes)'.format(item.filename, item.file_size))
            try:
                # Enough for ordinary KITTI timestamps/poses; cap abnormal entries.
                with archive.open(item) as stream:
                    raw = stream.read(2 * 1024 * 1024 + 1)
                truncated = len(raw) > 2 * 1024 * 1024
                lines = raw[:2 * 1024 * 1024].decode('utf-8', errors='replace').splitlines()
                emit('    Lines read: {}{}'.format(len(lines), ' (TRUNCATED)' if truncated else ''))
                preview = lines[:12] if PurePosixPath(item.filename).name == 'calib.txt' else lines[:3]
                for line in preview:
                    emit('    ' + line[:1000])
                if len(lines) > len(preview):
                    emit('    ...')
                    emit('    Last line read: ' + lines[-1][:1000])
                if PurePosixPath(item.filename).name == 'times.txt' and not truncated:
                    times = [float(line.strip()) for line in lines if line.strip()]
                    valid = all(math.isfinite(t) and t >= 0 for t in times)
                    deltas = [b - a for a, b in zip(times, times[1:])]
                    increasing = all(d > 0 for d in deltas)
                    emit('    Timestamps: {}; finite/nonnegative={}; strictly increasing={}'.format(
                        len(times), valid, increasing))
                    if deltas and valid and increasing:
                        emit('    Median interval={:.9g}s; nominal FPS={:.9g}'.format(
                            statistics.median(deltas), 1 / statistics.median(deltas)))
                    base = str(PurePosixPath(item.filename).parent)
                    for camera in ('image_2', 'image_3'):
                        group = images.get(base + '/' + camera)
                        if group is not None:
                            emit('    {} image count matches timestamps: {}'.format(camera, len(group) == len(times)))
            except Exception as exc:
                emit('    ERROR reading text: {}: {}'.format(type(exc).__name__, exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='ZIP files or directories containing ZIP files (nonrecursive)')
    parser.add_argument('--output', type=Path, default=Path('kitti_inspection_report.txt'))
    args = parser.parse_args()
    archives = []
    for path in args.paths:
        if path.is_dir():
            archives.extend(sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == '.zip'))
        elif path.is_file() and path.suffix.lower() == '.zip':
            archives.append(path)
        else:
            parser.error('Not a ZIP file or directory: {}'.format(path))
    archives = list(dict.fromkeys(p.resolve() for p in archives))
    if not archives:
        parser.error('No ZIP archives found')
    if args.output.resolve() in archives:
        parser.error('Output must not overwrite an input archive')
    errors = 0
    with args.output.open('w', encoding='utf-8', newline='\n') as report:
        def emit(line):
            report.write(line + '\n')
        emit('KITTI ARCHIVE INSPECTION REPORT v1')
        emit('Metadata, sample image headers and text previews only; no extraction.')
        emit('Image dimensions are sampled per folder, not checked for every frame.')
        emit('Full archive CRC/decode integrity has NOT been verified.')
        for archive in archives:
            print('Inspecting: {}'.format(archive.name), flush=True)
            try:
                inspect_archive(archive, emit)
            except Exception as exc:
                errors += 1
                emit('ARCHIVE ERROR: {}: {}'.format(type(exc).__name__, exc))
        emit('\nArchive-level failures: {}'.format(errors))
    print('Report saved: {}'.format(args.output.resolve()))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
