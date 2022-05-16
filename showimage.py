# coding=utf-8

import cv2
import numpy as np
import sensor_msgs.msg as msg

import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
      super().__init__('minimal_subscriber')
      self.subscription = self.create_subscription(
          msg.Image,
          'image',
          self.listener_callback,
          10)
      self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
      img = np.reshape(msg.data, (msg.height, msg.width, 3)).astype(np.uint8)
      cv2.imshow("Image", img)
      cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
