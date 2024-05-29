#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import WrenchStamped

# 初始化四个发布者，分别对应四个新话题
pub_FL_force = rospy.Publisher("/visual/FL_foot_contact/the_force_timestamped", WrenchStamped, queue_size=10)
pub_FR_force = rospy.Publisher("/visual/FR_foot_contact/the_force_timestamped", WrenchStamped, queue_size=10)
pub_RL_force = rospy.Publisher("/visual/RL_foot_contact/the_force_timestamped", WrenchStamped, queue_size=10)
pub_RR_force = rospy.Publisher("/visual/RR_foot_contact/the_force_timestamped", WrenchStamped, queue_size=10)

# 定义回调函数，给接收到的消息添加时间戳，并发布到新话题
def callback_FL_force(data):
    data.header.stamp = rospy.Time.now()
    pub_FL_force.publish(data)

def callback_FR_force(data):
    data.header.stamp = rospy.Time.now()
    pub_FR_force.publish(data)

def callback_RL_force(data):
    data.header.stamp = rospy.Time.now()
    pub_RL_force.publish(data)

def callback_RR_force(data):
    data.header.stamp = rospy.Time.now()
    pub_RR_force.publish(data)

if __name__ == '__main__':
    rospy.init_node('add_timestamp_node')

    # 初始化四个订阅者，分别订阅原始话题
    rospy.Subscriber("/visual/FL_foot_contact/the_force", WrenchStamped, callback_FL_force)
    rospy.Subscriber("/visual/FR_foot_contact/the_force", WrenchStamped, callback_FR_force)
    rospy.Subscriber("/visual/RL_foot_contact/the_force", WrenchStamped, callback_RL_force)
    rospy.Subscriber("/visual/RR_foot_contact/the_force", WrenchStamped, callback_RR_force)

    rospy.spin()
