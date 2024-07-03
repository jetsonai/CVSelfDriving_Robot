import rclpy
from rclpy.node import Node

#from std_msgs.msg import String
from custom_msgctl.msg import MsgCtl   


class MotorCtlPublisher(Node):

    def __init__(self):
        super().__init__('motorctl_publisher')
        #self.publisher_ = self.create_publisher(String, 'topic', 10)
        self.publisher_ = self.create_publisher(MsgCtl, 'MotorCtl', 10)     # CHANGE
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        #msg = String()
        msg = MsgCtl()                                           # CHANGE
        #msg.data = 'Hello World: %d' % self.i
        #msg.num = self.i  
        msg.speed_cmd = 0.3;
        msg.heading_cmd = 0.0;
        msg.dir_cmd = 0;
        msg.stop_cmd = 0;
        # CHANGE
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)
        #self.get_logger().info('Publishing: "%d"' % msg.num)  # CHANGE
        #self.get_logger().info('Publishing] speedCmd: "%0.2f" headingCmd: "%0.2f" dir_cmd: "%d"  stop_cmd: "%d"  ' % msg.speed_cmd, msg.heading_cmd, msg.dir_cmd , msg.stop_cmd )
        self.get_logger().info('Publishing: speedCmd: "%0.2f"' % msg.speed_cmd)
        self.get_logger().info('Publishing: heading_cmd: "%0.2f"' % msg.heading_cmd)
        self.get_logger().info('Publishing: dir_cmd: "%d"' % msg.dir_cmd)
        self.get_logger().info('Publishing: stop_cmd: "%d"' % msg.stop_cmd)

def main(args=None):
    rclpy.init(args=args)

    motorctl_publisher = MotorCtlPublisher()

    rclpy.spin(motorctl_publisher)

    motorctl_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()