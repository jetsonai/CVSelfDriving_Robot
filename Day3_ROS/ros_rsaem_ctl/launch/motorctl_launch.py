# motorctl_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_rsaem_ctl', # Package Name
            executable='motorctl_publisher', # Executable file
            emulate_tty=True),
    
        Node(
        package='ros_rsaem_ctl', # Package Name
        executable='rsaembot_motor', # Executable file
        output='screen',
        emulate_tty=True),
    ])