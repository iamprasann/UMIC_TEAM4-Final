#!/usr/bin/env python
 

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time

bridge = CvBridge()
#min area
min=30

#red[0] and green[1]
color_boundaries = [ ([0, 0, 220], [30, 30, 255]), ([0, 220, 0], [30, 255, 30])]


def image_callback(img_msg):
	rospy.loginfo(img_msg.header)
	try:
	    img = bridge.imgmsg_to_cv2(img_msg, "passthrough")
 
	    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    img = cv2.resize(img, (400,400))
	    classifier(img)
	
	except CvBridgeError, e:
		rospy.logerr("CvBridge Error: {0}".format(e))


def classifier(img):
    masks = []
    for (low, up) in color_boundaries:
        low=np.array(low, dtype='uint8')
        up=np.array(up, dtype='uint8')
        mask=cv2.inRange(img, low, up)
        op=cv2.bitwise_and(img, img, mask=mask)
        masks.append(op)

    red_zone = cv2.cvtColor(masks[0], cv2.COLOR_BGR2GRAY)
    green_zone = cv2.cvtColor(masks[1], cv2.COLOR_BGR2GRAY)

    _, red_zone = cv2.threshold(red_zone, 20, 255, cv2.THRESH_BINARY)
    _, green_zone = cv2.threshold(green_zone, 20, 255, cv2.THRESH_BINARY)

    _,contours_red, hierarchy = cv2.findContours(red_zone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _,contours_green, hierarchy = cv2.findContours(green_zone, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_red:
        if(cv2.contourArea(cnt)>2*min):
            print("Push the Red!")
    for cnt in contours_green:
        if(cv2.contourArea(cnt)>2*min):
            print("Grab the Green")
            

def listener():

	rospy.init_node('ball_detection',anonymous=True)
	rospy.Subscriber("/mybot/camera1/image_raw", Image, image_callback)
	rospy.spin()



### RUN ###
if __name__ == '__main__':
    listener()