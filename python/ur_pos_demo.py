#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

class URTestNode(Node):
    def __init__(self):
        super().__init__('ur_test_node')
        
        # Create publisher for forward position control
        self.publisher = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.timer = self.create_timer(1.0, self.send_command)  # 1 Hz
        self.command_index = 0

        # Define test positions for the joints (6 joints)
        self.test_commands = [
            [0.5, -0.5, 1.0, 0.0, -1.0, 0.5],
            [-0.5, 0.5, -1.0, 0.0, 1.0, -0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Reset position
        ]
        
    def send_command(self):
        # Get the next command
        if self.command_index < len(self.test_commands):
            joint_positions = self.test_commands[self.command_index]
            msg = Float64MultiArray()
            msg.data = joint_positions
            self.publisher.publish(msg)
            self.get_logger().info(f'Sent command: {joint_positions}')
            self.command_index += 1
        else:
            self.get_logger().info('Test sequence complete. Stopping node.')
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = URTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()