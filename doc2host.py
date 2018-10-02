#!/usr/bin/env python
import rospy
# ROS Image message
from sensor_msgs.msg import Image

INPUT_TOPIC = "/apollo/sensor/camera/traffic/image_long"
OUTPUT_TOPIC = "/steer_net"

def callback(img, pub):
    print("received")
    pub.publish(img)

def main():
    rospy.init_node('img_rep')
    pub = rospy.Publisher(OUTPUT_TOPIC, Image, queue_size= 10)
    rospy.Subscriber(INPUT_TOPIC, Image, callback, pub, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    main()
