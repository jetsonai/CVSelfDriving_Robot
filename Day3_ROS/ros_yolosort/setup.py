from setuptools import find_packages
import os
from glob import glob
from setuptools import setup

package_name = 'ros_yolosort'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))         
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetsonai',
    maintainer_email='jetsonai@jetswonai.co.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros_yolo_node = ros_yolosort.ros_yolo_node:main',
            'ros_yolo_dist = ros_yolosort.ros_yolo_dist:main',            
            'ros_yolo_sort_dist = ros_yolosort.ros_yolo_sort_dist:main',
            'ros_yolo_TTC = ros_yolosort.ros_yolo_TTC:main',
            'ros_yolo_driving = ros_yolosort.ros_yolo_driving:main',
        ],
    },
)

