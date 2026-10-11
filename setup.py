from setuptools import find_packages, setup

setup(
    name='kitti_dataset_tools',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/kitti_dataset_tools']),
        ('share/kitti_dataset_tools', ['package.xml', 'README.md', 'LICENSE']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alabex',
    maintainer_email='alabex@users.noreply.github.com',
    description='Independent KITTI odometry preparation, calibration conversion and ROS 2 playback.',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'kitti_player = kitti_dataset_tools.player:main',
        'generate_kitti_settings = kitti_dataset_tools.calibration:main',
        'prepare_kitti_dataset = kitti_dataset_tools.prepare:main',
        'inspect_kitti_archives = kitti_dataset_tools.inspect_archives:main',
    ]},
)
