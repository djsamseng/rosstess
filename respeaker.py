import argparse

import pyaudio
import wave
import numpy as np

import rclpy
from rclpy.node import Node
import std_msgs.msg


RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6 # change base on firmwares, 1_channel_firmware.bin as 1 or 6_channels_firmware.bin as 6
RESPEAKER_WIDTH = 2 # int16
ROS_MSG_TYPE = std_msgs.msg.Int16MultiArray
# run getDeviceInfo.py to get index
RESPEAKER_INDEX = 11  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"
WAVE_OUTPUT_VOICE_FILENAME = "output_voice.wav"


class RespeakerPublisher(Node):
  def __init__(self) -> None:
    super().__init__('RespeakerPublisher')
    self.pub = self.create_publisher(ROS_MSG_TYPE, 'audio', 10)
    self.timer = self.create_timer(100, self.timer_callback)

    self.p = pyaudio.PyAudio()

    self.stream = self.p.open(
      rate=RESPEAKER_RATE,
      format=self.p.get_format_from_width(RESPEAKER_WIDTH),
      channels=RESPEAKER_CHANNELS,
      input=True,
      input_device_index=RESPEAKER_INDEX,
      stream_callback=self.stream_callback)

  def on_shutdown(self):
    print("Shutdown")
    if self.stream.is_active():
      self.stream.stop_stream()

    self.stream.close()
    self.p.close()

  def timer_callback(self):
    pass

  def stream_callback(self, in_data, frame_count, time_info, status):
    data = in_data
    data = np.fromstring(data, dtype=np.int16)
    msg = ROS_MSG_TYPE()
    msg.data = [int(i) for i in data]
    self.pub.publish(msg)

    return None, pyaudio.paContinue


class RespeakerSubscriber(Node):
  def __init__(self):
    super().__init__('RespeakerSubscriber')
    self.subscription = self.create_subscription(
        ROS_MSG_TYPE,
        'audio',
        self.listener_callback,
        10)
    self.subscription  # prevent unused variable warning

  def listener_callback(self, msg):
    data = np.array(msg.data)
    for i in range(RESPEAKER_CHANNELS):
      channel_data = data[i::RESPEAKER_CHANNELS]
      print("channel:", i, channel_data.shape)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--publish",
    nargs="?",
    const=True,
    default=False,
    type=bool,
    dest="publish"
  )
  return parser.parse_args()

def main(args=None):
  mode_args = parse_args()
  rclpy.init(args=args)

  if mode_args.publish:
    node = RespeakerPublisher()
  else:
    node = RespeakerSubscriber()

  rclpy.spin(node)

if __name__ == "__main__":
  main()