/*
rsaem motor control
JetsonAI
Kate Kim
kate.brighteyes@gmail.com
*/

#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "custom_msgctl/msg/msg_ctl.hpp"


using namespace std::chrono_literals;


class MotoCtlPublisher : public rclcpp::Node
{
public:
  MotoCtlPublisher()
  : Node("MotoCtl_publisher"), count_(0)
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
    //motorctl_publisher_ = this->create_publisher<std_msgs::msg::String>(
    motorctl_publisher_ = this->create_publisher<custom_msgctl::msg::MsgCtl>(
	
      "MotorCtl", qos_profile);      
    timer_ = this->create_wall_timer(
      1s, std::bind(&MotoCtlPublisher::publish_motorctl_msg, this));
  }

private:
  void publish_motorctl_msg()
  {
   
    //auto msg = std_msgs::msg::String();
    auto msg = custom_msgctl::msg::MsgCtl();
    //msg.data = "motorctl: " + std::to_string(count_++);
    //RCLCPP_INFO(this->get_logger(), "Published message: '%s'", msg.data.c_str());
    
    msg.speed_cmd = 0.4;
    msg.heading_cmd = 0;
    msg.dir_cmd = 0;
    msg.stop_cmd = 0;
        
    RCLCPP_INFO(this->get_logger(), "Published speedCmd:%0.2f headingCmd:%0.2f dir_cmd:%d stop_cmd:%d", 
         msg.speed_cmd, msg.heading_cmd, msg.dir_cmd , msg.stop_cmd );
    motorctl_publisher_->publish(msg);
   
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<custom_msgctl::msg::MsgCtl>::SharedPtr motorctl_publisher_;
  size_t count_;
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MotoCtlPublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
