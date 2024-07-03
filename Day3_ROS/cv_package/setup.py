from setuptools import setup

package_name = 'cv_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pub_node = cv_package.pub_node:main',
            'lane_trace_cam_node = cv_package.lane_trace_cam_node:main',
            'lane_trace_cam_node2 = cv_package.lane_trace_cam_node2:main',
            'lane_trace_drive_node = cv_package.lane_trace_drive_node:main'
        ],
    },
)
