# kitti_dataset_tools

独立的 ROS 2 KITTI Odometry 工具包（ament_python，已在 Jazzy 验证）。
不依赖 ORB-SLAM3、Pangolin、PCL 或原 SLAM 源码，可复制本目录到任意 ROS 2 工作空间的 `src/`。

## 构建

在工作空间根目录运行：

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src/kitti_dataset_tools --ignore-src -r -y
colcon build --packages-select kitti_dataset_tools
source install/setup.bash
```

## 命令

| 命令 | 功能 |
| --- | --- |
| `prepare_kitti_dataset` | 解压并合并图像、标定、位姿和可选 Velodyne，CRC 校验及断点续跑 |
| `inspect_kitti_archives` | 检查 ZIP 目录及元数据，生成检查报告，不解压整个数据集 |
| `generate_kitti_settings` | 从 KITTI `P2/P3`、图像尺寸和时间戳生成 ORB-SLAM3 相机 YAML |
| `kitti_player` | 向 ROS 2 发布单目/双目图像、可选激光点云和 `/clock` |

生成配置是对已有标定的格式转换，不执行相机或雷达外参标定。
配置输出适配 ORB-SLAM3，但转换过程无需安装或运行 ORB-SLAM3。
每个命令支持 `--help`。

### 整理数据

```bash
ros2 run kitti_dataset_tools prepare_kitti_dataset \
  --archives-dir /home/alabex/下载/kitti_data \
  --output /home/alabex/dataset/kitti --require-velodyne

ros2 run kitti_dataset_tools inspect_kitti_archives \
  /home/alabex/下载/kitti_data --output /tmp/kitti_inspection.txt
```

整理工具要求 `data_odometry_color.zip`、`data_odometry_calib.zip`；
可选 `data_odometry_poses.zip`、`devkit_odometry.zip` 和 `data_odometry_velodyne.zip`。
`--require-velodyne` 要求激光压缩包存在；`--sequences 00 01` 限定整理范围。
已存在且 CRC 一致的文件会跳过，覆盖不同内容需显式指定 `--overwrite`。

```text
kitti/sequences/00/
├── image_2/000000.png
├── image_3/000000.png
├── velodyne/000000.bin
├── calib.txt
└── times.txt
```

### 生成相机配置

```bash
ros2 run kitti_dataset_tools generate_kitti_settings \
  --dataset-root /home/alabex/dataset/kitti --sequence 00 \
  --settings-output /home/alabex/dataset/kitti/kitti_00_color.yaml
```

无需启动 ROS 节点。兼容 `kitti_player --settings-output ... --prepare-only`。

### 发布图像和激光

```bash
ros2 run kitti_dataset_tools kitti_player \
  --dataset-root /home/alabex/dataset/kitti --sequence 00 \
  --publish-velodyne --publish-clock --wait-for-subscribers 0
```

默认发布话题：

| 话题 | 消息类型 | frame |
| --- | --- | --- |
| `/kitti/left/image_raw` | `sensor_msgs/msg/Image`，bgr8 | `kitti_left_optical_frame` |
| `/kitti/right/image_raw` | `sensor_msgs/msg/Image`，bgr8 | `kitti_right_optical_frame` |
| `/kitti/velodyne_points`（显式启用） | `sensor_msgs/msg/PointCloud2`，float32 XYZI | `kitti_velodyne` |
| `/clock`（显式启用） | `rosgraph_msgs/msg/Clock` | — |

点云与同帧图像的时间戳相同；速度只改变发送节奏。支持 `--speed 0.5`、
`--start-frame 100 --max-frames 500`、`--mode mono`、`--reliability best_effort`。
默认 QoS 为 reliable、volatile、depth 10。话题通过 `--left-topic`、`--right-topic`、
`--velodyne-topic` 配置。更改 `--velodyne-frame` 只修改名称，不变换坐标。

默认等待图像订阅者最多 60 秒，不等待激光订阅者。上述示例通过
`--wait-for-subscribers 0` 直接播放；与 SLAM 联用时可去掉此选项，并在 SLAM
设置 `use_sim_time:=true`。未开启 `/clock` 时使用系统时间起点加数据集时间间隔。

RViz 查看原始点云时将 Fixed Frame 设为 `kitti_velodyne`。
本包不发布 TF、CameraInfo 或 IMU，不进行激光运动补偿、位姿估计或地图融合。
仅支持 KITTI Odometry，非 KITTI Raw/KITTI-360。

## 测试

```bash
colcon test --packages-select kitti_dataset_tools --event-handlers console_direct+
colcon test-result --verbose
```

测试使用模拟数据，覆盖标定转换、数据整理和 ROS 2 同步发布，不需要真实数据或 SLAM。

## 与ORB项目的关系

旧命令 `ros2 run orbslam3_dense_ros2 kitti_player` 及旧 Python 脚本仅作为兼容入口，
要求先构建并 source 本包。新项目应直接使用上述 `ros2 run kitti_dataset_tools ...` 命令。
SLAM 本身可以单独构建，使用其他相机或数据发布器不需要本包。

### 一次生成所有序列的相机 YAML

```bash
ros2 run kitti_dataset_tools generate_kitti_settings \
  --dataset-root /home/alabex/dataset/kitti \
  --all-sequences \
  --output-dir /home/alabex/dataset/kitti/config
```

扫描 `sequences/00` 到 `sequences/21` 中实际存在的目录，分别读取各自的标定、
图像尺寸和时间戳，输出 `kitti_00_color.yaml`、`kitti_01_color.yaml` 等。
不指定 `--output-dir` 时默认存入数据集根目录下的 `config/`。
不存在的序列不生成；存在但数据不完整的序列会报错，继续处理其他序列，最终返回非零状态。
重复执行会覆盖同名 YAML。单序列 `--sequence ... --settings-output ...` 用法仍然有效。

## RGB-D 深度融合与建图

`depth_fusion.py` 是不依赖 ROS 的计算模块，`player.py` 负责读取、调度与发布。
`--mode rgbd` 在原双目话题基础上增加激光和深度，始终同时发布四路数据：
`/kitti/left/image_raw`、`/kitti/right/image_raw`、`/kitti/velodyne_points`、
`/kitti/depth/image_raw`。无需额外指定 `--publish-velodyne`。
深度为 32FC1（米，0 为无效），与左图尺寸和光学坐标系一致；四路使用同帧时间戳。
深度来源选项只影响深度计算，不会关闭右图或激光话题，因此 RGB-D 模式始终要求点云文件。

生成 RGB-D 专用配置（不会覆盖已有的 `_color.yaml`）：

```bash
ros2 run kitti_dataset_tools generate_kitti_settings \
  --dataset-root /home/alabex/dataset/kitti --all-sequences --mode rgbd \
  --output-dir /home/alabex/dataset/kitti/config
```

终端一启动原 SLAM（使用 RGB-D YAML，浮点米制深度因子为 1）：

```bash
source /home/alabex/slam_ros2/install/setup.bash
ros2 launch orbslam3_dense_ros2 rgbd.launch.py \
  settings_file:=/home/alabex/dataset/kitti/config/kitti_00_rgbd.yaml \
  rgb_topic:=/kitti/left/image_raw depth_topic:=/kitti/depth/image_raw \
  camera_frame:=kitti_left_optical_frame use_sim_time:=true image_reliability:=reliable \
  experiment_output_dir:=/home/alabex/slam_ros2/results/kitti_00_rgbd
```

终端二播放：

```bash
source /home/alabex/slam_ros2/install/setup.bash
ros2 run kitti_dataset_tools kitti_player \
  --dataset-root /home/alabex/dataset/kitti --sequence 00 \
  --mode rgbd --depth-source fused --publish-clock --speed 0.25
```

默认等待左图和深度订阅者，不要求右图或激光有订阅者，四路仍全部发布；仅观察话题时加 `--wait-for-subscribers 0`。
`--depth-topic` 修改深度话题。`--depth-source stereo|lidar|fused` 提供对照：

- stereo：SGBM，唯一性/散斑过滤及双向视差一致性检查；深度计算本身不使用点云，但 RGB-D 播放仍读取并发布激光。
- lidar：原始标定 Tr、完整 P2 投影，输出左相机 Z；同像素保留最近返回。
- fused（默认）：以有效双目深度为基础，在激光落点处采用激光深度；若激光比
  双目前景深超过 15%，拒绝该返回以减轻遮挡穿透。双目无效处可采用有效激光。

这是一种保守的像素级融合基线，不是稠密深度补全或紧耦合定位。
不扩散稀疏激光点、不把无效区域强行填满，也不保证所有激光遮挡都被消除。
没有运动补偿和动态物体分割；双目匹配使用 KITTI 已校正图像的近似水平极线。
计算深度考虑 P2/P3 的投影平移，避免把雷达距离当作相机 Z。

默认有效范围 0.5–80 米，可设 `--min-depth`、`--max-depth`。
SGBM 默认 `--num-disparities 128 --block-size 5`；视差数量必须为 16 的倍数，
块大小为 3–21 的奇数。视差上限会限制近距离双目覆盖，可按需求增加。
CPU 计算可能低于 10 Hz；降低 `--speed` 保留数据集时间戳并给 SLAM 留出处理时间。

**现有 SLAM 的 `PointClouds.cc` 仍有 5 米稠密建图过滤阈值。** 本工具包不修改
SLAM 算法，因此远处深度可以发布并用于跟踪，但该稠密地图输出仍会被截断。
整段地图精度、轨迹误差和比纯双目是否改善需另做对照评估。
