#!/usr/bin/env python3
"""Publish rectified KITTI color pairs and optional Velodyne clouds; derive ORB settings from P2/P3."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from .calibration import read_calibration, write_settings


def read_sequence(directory):
    directory = Path(directory).expanduser().resolve()
    left = sorted((directory / 'image_2').glob('*.png'))
    right = sorted((directory / 'image_3').glob('*.png'))
    if not left or [p.name for p in left] != [p.name for p in right]:
        raise ValueError('image_2 and image_3 must contain matching, nonempty PNG frame lists')
    timestamps = np.atleast_1d(np.loadtxt(directory / 'times.txt', dtype=np.float64))
    if timestamps.ndim != 1 or len(timestamps) != len(left):
        raise ValueError('times.txt must contain exactly one timestamp per image pair')
    if not np.isfinite(timestamps).all() or (timestamps < 0).any() or (np.diff(timestamps) <= 0).any():
        raise ValueError('Timestamps must be finite, nonnegative and strictly increasing')
    expected = [f'{i:06d}.png' for i in range(len(left))]
    if [p.name for p in left] != expected:
        raise ValueError('Expected contiguous KITTI frames starting at 000000.png; use --start-frame to skip frames')
    return left, right, timestamps


def read_velodyne_sequence(directory, images):
    scans = sorted((Path(directory) / 'velodyne').glob('*.bin'))
    if [p.stem for p in scans] != [p.stem for p in images]:
        raise ValueError('velodyne must contain one matching .bin file per image frame')
    for path in scans:
        size = path.stat().st_size
        if size == 0 or size % 16:
            raise ValueError(f'Invalid Velodyne file (expected nonempty float32 XYZI records): {path}')
    return scans


def read_velodyne_message(path, stamp, frame_id):
    from sensor_msgs.msg import PointCloud2, PointField
    data = Path(path).read_bytes()
    if not data or len(data) % 16:
        raise ValueError(f'Invalid Velodyne file: {path}')
    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = 1
    message.width = len(data) // 16
    message.fields = [PointField(name=name, offset=offset, datatype=PointField.FLOAT32, count=1)
                      for name, offset in zip(('x', 'y', 'z', 'intensity'), (0, 4, 8, 12))]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = len(data)
    message.is_dense = bool(np.isfinite(np.frombuffer(data, dtype='<f4')).all())
    message.data = data
    return message


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--sequence-dir', type=Path, help='Path to sequences/00 (or another sequence)')
    source.add_argument('--dataset-root', type=Path, help='Dataset root containing sequences/')
    parser.add_argument('--sequence', help='Sequence number 00..21; required with --dataset-root')
    parser.add_argument('--settings-output', type=Path, help='Write calibrated ORB-SLAM3 YAML for the selected mode')
    parser.add_argument('--prepare-only', action='store_true', help='Validate data and write settings without publishing')
    parser.add_argument('--left-topic', default='/kitti/left/image_raw')
    parser.add_argument('--right-topic', default='/kitti/right/image_raw')
    parser.add_argument('--publish-velodyne', action='store_true', help='Publish synchronized velodyne/*.bin as PointCloud2')
    parser.add_argument('--velodyne-topic', default='/kitti/velodyne_points')
    parser.add_argument('--velodyne-frame', default='kitti_velodyne', help='Native LiDAR frame; changing the name does not transform points')
    parser.add_argument('--mode', choices=('stereo', 'mono', 'rgbd'), default='stereo',
                        help='Publish stereo, mono, or all four streams: stereo images, LiDAR and depth')
    parser.add_argument('--depth-topic', default='/kitti/depth/image_raw')
    parser.add_argument('--depth-source', choices=('fused', 'stereo', 'lidar'), default='fused')
    parser.add_argument('--min-depth', type=float, default=0.5)
    parser.add_argument('--max-depth', type=float, default=80.0)
    parser.add_argument('--num-disparities', type=int, default=128)
    parser.add_argument('--block-size', type=int, default=5)
    parser.add_argument('--reliability', choices=('reliable', 'best_effort'), default='reliable',
                        help='Image and point cloud QoS reliability; reliable supports reliable and best-effort subscribers')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed; 0.5 is half real time')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--wait-for-subscribers', type=float, default=60.0, help='Timeout in seconds; 0 disables waiting')
    parser.add_argument('--publish-clock', action='store_true', help='Publish dataset time to /clock; set SLAM use_sim_time:=true')
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.speed) or args.speed <= 0:
            raise ValueError('--speed must be positive')
        if not np.isfinite(args.wait_for_subscribers) or args.wait_for_subscribers < 0:
            raise ValueError('--wait-for-subscribers must be nonnegative')
        if args.prepare_only and not args.settings_output:
            raise ValueError('--prepare-only requires --settings-output')
        if args.dataset_root:
            if args.sequence is None or not args.sequence.isdecimal() or not 0 <= int(args.sequence) <= 21:
                raise ValueError('--dataset-root requires --sequence between 00 and 21')
            directory = args.dataset_root / 'sequences' / f'{int(args.sequence):02d}'
        else:
            if args.sequence is not None:
                raise ValueError('Use --sequence with --dataset-root, not --sequence-dir')
            directory = args.sequence_dir
        directory = directory.expanduser().resolve()
        left, right, timestamps = read_sequence(directory)
        read_calibration(directory)
        publish_lidar = args.publish_velodyne or args.mode == 'rgbd'
        needs_lidar = publish_lidar
        scans = read_velodyne_sequence(directory, left) if needs_lidar else None
        fusion = None
        if args.mode == 'rgbd':
            from .depth_fusion import DepthFusion, FusionConfig
            fusion = DepthFusion(directory, FusionConfig(min_depth=args.min_depth, max_depth=args.max_depth,
                                num_disparities=args.num_disparities, block_size=args.block_size), args.depth_source)
        if publish_lidar and (not args.velodyne_frame.strip() or args.velodyne_frame.startswith('/')):
            raise ValueError('--velodyne-frame must be nonempty and must not start with /')
        print(f'Sequence: {directory}; {len(left)} stereo pairs; speed: {args.speed}x', flush=True)
        if not 0 <= args.start_frame < len(left) or (args.max_frames is not None and args.max_frames < 1):
            raise ValueError('Invalid --start-frame or --max-frames')
        if args.settings_output:
            write_settings(args.settings_output, directory, left[0], right[0], timestamps, mode='rgbd' if args.mode == 'rgbd' else 'stereo')
        if args.prepare_only:
            return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))

    # Keep validation/config generation usable without a running ROS graph.
    import rclpy
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, PointCloud2
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time
    rclpy.init(args=[])
    node = rclpy.create_node('kitti_odometry_player')
    image_qos = QoSProfile(depth=10,
                           reliability=(ReliabilityPolicy.RELIABLE if args.reliability == 'reliable'
                                        else ReliabilityPolicy.BEST_EFFORT),
                           durability=DurabilityPolicy.VOLATILE)
    topics = (args.left_topic, args.right_topic) if args.mode != 'mono' else (args.left_topic,)
    publishers = [node.create_publisher(Image, topic, image_qos) for topic in topics]
    cloud_pub = node.create_publisher(PointCloud2, args.velodyne_topic, image_qos) if publish_lidar else None
    depth_pub = node.create_publisher(Image, args.depth_topic, image_qos) if fusion is not None else None
    clock_pub = node.create_publisher(Clock, '/clock', qos_profile_sensor_data) if args.publish_clock else None
    try:
        node.get_logger().info(f'Publishing {", ".join(topics)} (bgr8, {args.reliability}, volatile, depth=10)')
        if cloud_pub is not None:
            node.get_logger().info(f'Publishing {args.velodyne_topic} (PointCloud2 XYZI, frame={args.velodyne_frame}); LiDAR subscriber is optional')
        if depth_pub is not None:
            node.get_logger().info(f'Publishing {args.depth_topic}: 32FC1 metres, source={args.depth_source}')
        # RGB-D consumers need only left + depth; right and LiDAR still publish.
        required_publishers = [publishers[0], depth_pub] if depth_pub is not None else publishers
        if args.wait_for_subscribers:
            node.get_logger().info('Waiting for subscribers on enabled image topics...')
        deadline = time.monotonic() + args.wait_for_subscribers
        while args.wait_for_subscribers and any(pub.get_subscription_count() == 0 for pub in required_publishers):
            if not rclpy.ok():
                return 0
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for image subscribers; start the matching SLAM launch first')
            rclpy.spin_once(node, timeout_sec=0.1)
        start = args.start_frame
        end = min(len(left), start + args.max_frames) if args.max_frames else len(left)
        base_ns = 0 if args.publish_clock else node.get_clock().now().nanoseconds - round(timestamps[start] * 1e9)
        wall_start = time.monotonic()
        expected_shape = None
        for i in range(start, end):
            due = wall_start + (timestamps[i] - timestamps[start]) / args.speed
            while rclpy.ok() and time.monotonic() < due:
                remaining = max(0.0, due - time.monotonic())
                rclpy.spin_once(node, timeout_sec=min(0.02, remaining))
            if not rclpy.ok():
                break
            images = [cv2.imread(str(paths[i]), cv2.IMREAD_COLOR) for paths in (left, right)]
            if any(image is None for image in images) or images[0].shape != images[1].shape:
                raise RuntimeError(f'Unreadable or mismatched image pair at frame {i}')
            if expected_shape is None:
                expected_shape = images[0].shape
            if images[0].shape != expected_shape:
                raise RuntimeError(f'Image dimensions change at frame {i}')
            nanoseconds = base_ns + round(timestamps[i] * 1e9)
            stamp = Time(sec=int(nanoseconds // 1000000000), nanosec=int(nanoseconds % 1000000000))
            cloud = read_velodyne_message(scans[i], stamp, args.velodyne_frame) if publish_lidar else None
            depth = None
            if fusion is not None:
                points = np.fromfile(scans[i], dtype='<f4').reshape(-1, 4) if scans is not None else None
                depth = fusion.compute(images[0], images[1], points)['depth']
            if clock_pub:
                clock_pub.publish(Clock(clock=stamp))
            for pub, image, frame in zip(publishers, images, ('kitti_left_optical_frame', 'kitti_right_optical_frame')):
                message = Image()
                message.header.stamp = stamp
                message.header.frame_id = frame
                message.height, message.width = image.shape[:2]
                message.encoding = 'bgr8'
                message.step = message.width * 3
                message.data = image.tobytes()
                pub.publish(message)
            if depth_pub is not None:
                message = Image()
                message.header.stamp = stamp
                message.header.frame_id = 'kitti_left_optical_frame'
                message.height, message.width = depth.shape
                message.encoding = '32FC1'
                message.is_bigendian = False
                message.step = message.width * 4
                message.data = depth.astype('<f4', copy=False).tobytes()
                depth_pub.publish(message)
            if cloud_pub is not None:
                cloud_pub.publish(cloud)
            rclpy.spin_once(node, timeout_sec=0.0)
            if (i - start) % 100 == 0:
                node.get_logger().info(f'Published {i - start + 1}/{end - start} pairs ({100 * (i - start + 1) / (end - start):.1f}%), frame {i}')
        # Give DDS time to send the last frame pair before destroying the publishers.
        until = time.monotonic() + 0.5
        while rclpy.ok() and time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.05)
        return 0
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        node.get_logger().error(str(error))
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
