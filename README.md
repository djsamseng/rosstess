# ROS

```bash
source /opt/ros/galactic/setup.bash
export ROS_DOMAIN_ID=9
```

## Webcam
```bash
ros2 run image_tools showimage
python3 showimage.py # Alternatively
```

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext # get sizes
ros2 run image_tools cam2image --ros-args -p width:=640 -p height:=360 -p device_id:=0
```

## Audio
```bash
python3 respeaker.py --publish
python3 respeaker.py
```