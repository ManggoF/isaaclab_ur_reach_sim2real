""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "base_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "0.114605",
                "--y",
                "0.274584",
                "--z",
                "0.177882",
                "--qx",
                "0.055145",
                "--qy",
                "0.132667",
                "--qz",
                "0.161882",
                "--qw",
                "0.976296",
                # "--roll",
                # "0.0674077",
                # "--pitch",
                # "0.280564",
                # "--yaw",
                # "0.319112",
            ],
        ),
    ]
    return LaunchDescription(nodes)
