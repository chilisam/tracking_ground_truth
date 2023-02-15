from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="tracking_ground_truth",
            executable="tracking_gt",
            name="tracking_gt_node",
            output="screen",
            emulate_tty=True,
            parameters= [{"only_gt": False}]
        )
    ])
