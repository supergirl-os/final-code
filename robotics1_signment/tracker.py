from geometry_msgs.msg import Twist
from turtlesim.srv import Spawn
from turtlesim.msg import Pose
from rclpy.node import Node
from .pid import PID
from .utils import *

import random
import rclpy
import math
class pose:
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y
class turtlesTracker(Node):
    def __init__(self,text=None,flag=False):
        if text is not None:
            name = text
        else:
            name = 'leader'
        self.flag = flag
        super().__init__("tracker_"+name)
        # 现有位置信息 ---------
        # 现有位置信息:  X   Y  
        self.places = [0., 0.]
        self.thetas = 0.
        self.orginx = 5.5
        self.orginy = 5.5
        # 设置速度信息 -------------------------
        # 设置速度信息:  X   Y
        self.speeds = [1., 1.]
        # 初始化PID信息 ------------------------
        self.pids_x = PID(1.5, 0., 0.)
        self.pids_y = PID(1.5, 0., 0.)
        # 需要新建两只乌龟 --------------------------------
        self.serice = self.create_client(Spawn, 'spawn')
        while not self.serice.wait_for_service(timeout_sec=5):
            self.get_logger.info("Create Client Failed")
        self.requ_1 = Spawn.Request()
        if text is not None:
            self.newone(name)
        # 初始化话题信息 ---------------------------------------------------------------
        self.read_1 = self.create_subscription(Pose, '/'+name+'/pose', self.read_1, 10)
        self.push_1 = self.create_publisher(Twist, '/'+name+'/cmd_vel', 100)
        self.update()
    # 读取信息 -----------------
    def read_1(self,msg):
        self.places[0] = msg.x
        self.places[1] = msg.y
        self.thetas    = msg.theta
        # 乌龟如果撞墙了 ---------------------------------------
        if msg.y>11.08 or msg.x>11.08 or msg.x<=0 or msg.y<=0: 
            if not self.flag:
                k1 = self.pids_x.update((self.places[0]-self.orginx)*(-1.))*(-1.)
                k2 = self.pids_y.update((self.places[1]-self.orginy)*(-1.))*(-1.)
                # self.speeds[0] =  k1 if k1!=0 else 0.00000001
                # self.speeds[1] =  k2 if k2!=0 else 0.00000001
                self.speeds[0] =  random.randint(-20, 20)/10
                self.speeds[1] =  random.randint(-20, 20)/10
                self.get_logger().info("I hit wall")
                self.update()
    def newone(self,name):
        p = Pose(x=float(random.randint(1, 10)), 
                 y=float(random.randint(1, 10)))
        self.orginx = p.x
        self.orginy = p.y
        self.requ_1.x = p.x
        self.requ_1.y = p.y
        self.requ_1.name = name
        self.serice.call_async(self.requ_1)
    # 设置乌龟速度和方向 ----------------------
    def update(self,flag=None): 
        twist1 = Twist()
        try:
            twist1.linear.x = float(self.speeds[0])
            twist1.linear.y = float(self.speeds[1])
        except AttributeError:
            print(type(self.speeds[0]))
        # twist1.angular.x = math.atan(self.speeds[1]/(0.000001 if self.speeds[0]==0 else self.speeds[0]))
        self.push_1.publish(twist1)
        #self.get_logger().info("{} {}".format(self.speeds[0],self.speeds[1]))

def main(args=None):
    rclpy.init(args=args)
    # 初始化类对象 ---------------------
    track1 = turtlesTracker('turtle1')
    track2 = turtlesTracker('turtle2',True)
    track3 = turtlesTracker('turtle3',True)
    # 初始化数据 -----------------------------------
    main_loops = 0
    track1.speeds[0] = random.randint(-20, 20)/10
    track1.speeds[0] = random.randint(-20, 20)/10
    track1.update()
    # 主循环 ----------
    while rclpy.ok():
        main_loops+=1
        rclpy.spin_once(track1)
        rclpy.spin_once(track2)
        rclpy.spin_once(track3)
        # 计算速度PID和角度信息 -------------------------------------------------------------------------------
        track2.speeds[0] = track2.pids_x.update((track1.places[0]-1)-track2.places[0])
        track2.speeds[1] = track2.pids_y.update((track1.places[1]-1)-track2.places[1])
        track3.speeds[0] = track3.pids_x.update((track1.places[0]-1)-track3.places[0])
        track3.speeds[1] = track3.pids_y.update((track1.places[1]+1)-track3.places[1])
        # track2.speeds[0] = ((track1.places[0]-1)-track2.places[0])*(-1)
        # track2.speeds[1] = ((track1.places[1]-1)-track2.places[1])*(-1)
        # track3.speeds[0] = ((track1.places[0]-1)-track3.places[0])*(-1)
        # track3.speeds[1] = ((track1.places[1]+1)-track3.places[1])*(-1)
        if main_loops % 100 ==0:
            track1.update()
        # if main_loops == 1000:
        #     track1.speeds[0] = random.randint(-20, 20)/10
        #     track1.speeds[0] = random.randint(-20, 20)/10
        #     track1.update()
        track2.update()
        track3.update()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
