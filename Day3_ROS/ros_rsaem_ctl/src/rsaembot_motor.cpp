#include <cstdio>
   
#include "rsaembot_motor.h"
#include <string>
#include <sstream>

#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include "cansocket.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "custom_msgctl/msg/msg_ctl.hpp"

using namespace std::chrono_literals;

using namespace std;

static volatile bool gRun = true;

void init_can(void);

void rsaembot_setup(void);
//void aibot_loop(void);
void *rsaembot_loop(void* data);
void close_rsaembot(void);

void sig_int_handler(int sig);
void set_signal(void);

int sendCanTx2(float speed, int dir, float angle, int aibotgo);
void motor_contol_test(void);
void motor_contol(float speedCmd, float headingCmd, uint32_t dirCmd, uint32_t stopCmd);

bool setup_end = false;

using std::placeholders::_1;



class MotoCtlSubscriber : public rclcpp::Node
{
public:
  MotoCtlSubscriber()
  : Node("MotoCtlSubscriber")
  {
    auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
//    motorctl_subscriber_ = this->create_subscription<std_msgs::msg::String>(
    motorctl_subscriber_ = this->create_subscription<custom_msgctl::msg::MsgCtl>(
      "MotorCtl",
      qos_profile,
      std::bind(&MotoCtlSubscriber::subscribe_topic_message, this, _1));
      count = 0;
  }
  int count;

private:
  //void subscribe_topic_message(const std_msgs::msg::String::SharedPtr msg) const
  void subscribe_topic_message(const custom_msgctl::msg::MsgCtl::SharedPtr msg) 
  { 
    RCLCPP_INFO(this->get_logger(), "recieve speedCmd:%0.2f headingCmd:%0.2f dir_cmd:%d stop_cmd:%d", 
         msg->speed_cmd, msg->heading_cmd, msg->dir_cmd , msg->stop_cmd );
 
 motor_contol(msg->speed_cmd, msg->heading_cmd, msg->dir_cmd, msg->stop_cmd);

  }
  rclcpp::Subscription<custom_msgctl::msg::MsgCtl>::SharedPtr motorctl_subscriber_;

};


int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  printf("hello world ros_rsaem_ctl package\n");

  set_signal();

  gRun = true;
  
  motorLeft = 1.0;
  motorRight = 1.0;
  printf("[main] motorLeft %f motorRight %f\n",motorLeft, motorRight);

  rsaembot_setup();

  rclcpp::init(argc, argv);

#if 1
   auto node = std::make_shared<MotoCtlSubscriber>();
   rclcpp::spin(node); 
#endif 
  
#if 1

  pthread_t thread_t2;
  if (pthread_create(&thread_t2, NULL, rsaembot_loop, 0) < 0)
  {
    printf("thread2 create error:");
    exit(0);
  }

  int status2;
  pthread_join(thread_t2, (void **)&status2);
  printf("Thread2 End %d\n", status2);
#endif

  close_rsaembot();  
  rclcpp::shutdown();
  return 0;
}

void rsaembot_setup(void)
{
  init_can();

  setup_end = true;
}

void motor_contol_test(void)
{
    float speedCmd = 0.5;
    float headingCmd = 0.1; 
    uint32_t dirCmd = 0;
    uint32_t stopCmd = 0;
    if(speedCmd > 0) {
	  dirCmd = 0; 
    } else if (speedCmd < 0) {
	  dirCmd = 1; //0;
    } 
    speedCmd = abs(speedCmd);
    if (speedCmd == 0 && headingCmd==0) {
      stopCmd = 1;
    } 
  printf("speedCmd:%0.2f headingCmd:%0.2f dirCmd:%d stopCmd:%d\n", speedCmd, headingCmd, dirCmd, stopCmd);	   
  sendCanTx2( speedCmd, dirCmd, headingCmd, stopCmd);

}

void motor_contol(float speedCmd, float headingCmd, uint32_t dirCmd, uint32_t stopCmd)
{
    if(speedCmd > 0) {
	  dirCmd = 0; 
    } else if (speedCmd < 0) {
	  dirCmd = 1; //0;
    } 
    speedCmd = abs(speedCmd);
    if (speedCmd == 0 && headingCmd==0) {
      stopCmd = 1;
    } 
  printf("speedCmd:%0.2f headingCmd:%0.2f dirCmd:%d stopCmd:%d\n", speedCmd, headingCmd, dirCmd, stopCmd);	   
  sendCanTx2( speedCmd, dirCmd, headingCmd, stopCmd);

}

/*******************************************************************************
* Loop function
*******************************************************************************/

void *rsaembot_loop(void* data)
{
  printf("<rsaembot_loop>\n");
  rclcpp::Rate loop_rate(30);

    while (gRun && rclcpp::ok())
    {
        printf("rsaembot_loop \n");
        //motor_contol_test();
        loop_rate.sleep();              
    }
    printf("end loop thread..\n");
    motor_contol(0,0,0,1);
    sleep(1);
    close_port();

}

void set_signal(void)
{
  struct sigaction action = {};
  action.sa_handler = sig_int_handler;

  sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
  sigaction(SIGINT, &action, NULL);  // Ctrl-C
  sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
  sigaction(SIGABRT, &action, NULL); // abort() called.
  sigaction(SIGTERM, &action, NULL); // kill command	
}

void sig_int_handler(int sig)
{
    
    (void)sig;
    gRun = false;
    //motor_contol(0,0,0,1);
    //close_port();
}

void init_can(void)
{
  if(open_port("can0") < 0)
  {
    printf("open can error:");
    exit(0);

  }
  printf("can0 opened.\n");	
}

void close_rsaembot(void)
{
  printf("close_rsaembot\n");
  close_port();	
}

#define MotorMin 130 //55
#define MotorMax 255
#define InputMin 0
#define InputMax 1.0

int motor_map( float x ) 
{
    if(x >= 1) x = 1;

    int out_min = MotorMin;
    int out_max = MotorMax;
    float in_min= InputMin;
    float in_max= InputMax;
    int ret =(x-in_min)*(out_max-out_min)/(in_max-in_min)+out_min ;
    // For debugging:
    //printf("MAPPED %f to: %d\n", x, ret);
    return ret ;
}

float getmin(float f1, float f2)
{
    if(f1 <= f2)
       return f1;
    else
       return f2;
}

float getmax(float f1, float f2)
{
    if(f1 >= f2)
       return f1;
    else
       return f2;
}

int sendCanTx2(float speed, int dir, float angle, int aibotgo)
{
    struct can_frame canmsg, res;
    static float pr_speed2, pr_angle;
    static int pr_dir2, pr_aibotgo2;
    int speedL, speedR;

    if(pr_speed2==speed && pr_dir2==dir && pr_angle==angle && pr_aibotgo2==aibotgo) 
    {
        return -1;
    }
    else 
    {
      pr_speed2=speed;
      pr_dir2=dir;
      pr_angle=angle;
      pr_aibotgo2=aibotgo;
    }
	
	/*if(aibotgo==1) {
		printf("<RunningMode:(%d)(%d)(%d)(%d) >>>>>>>> STOP!!\n", 
		                    RunningMode_F, RunningMode_B, SaftyMode_F, SaftyMode_B);
	}*/
	
    printf("[sendCanTx2] speed:%0.2f angle:%0.2f dir:%d aibotgo:%d\n", speed, angle, dir, aibotgo);
    float left_x = 0;
    float right_x = 0;
    dirL = dirR = dir;
    if(speed == 0 && angle != 0) {
        if(angle > 0) { //right side 
            printf("Turn right side");
            dirL = 0;
            dirR = 1;	  
        } 
        else if(angle < 0) { //left side
            printf("Turn left side");
            dirL = 1;
            dirR = 0;	  
        }
        left_x = right_x = 0.1;
    }

    if(speed > 0) 
	{   
		static const float steering_dgain = 0.1;
		static const float speed_gain = 1.0;	
		static int angle_last;
        float pid = angle * steering_dgain + (angle-angle_last) * steering_dgain;
        angle_last = angle;
	
        float speed_value = speed * speed_gain;
        float steering_value = pid;
    
        left_x = getmax(getmin(speed_value + steering_value, 1.0), 0.0);
        right_x = getmax(getmin(speed_value - steering_value, 1.0), 0.0);
    }

    left_x = left_x * motorLeft;
    right_x = right_x * motorRight;
    printf("motorLeft %f motorRight %f left_x %f right_x %f\n",motorLeft, motorRight,left_x,right_x);

    speedL = motor_map(left_x);
    speedR = motor_map(right_x);
    printf("speedL %d speedR %d dirL %d dirR %d aibotgo %d Running(%d)(%d) Safety(%d)(%d) ", 
	            speedL,speedR, dirL, dirR, aibotgo, RunningMode_F, 
				RunningMode_B, SaftyMode_F, SaftyMode_B);
    memset((void*)&canmsg, 0x00, sizeof(struct can_frame));
    canmsg.can_id = MSG_AIBOT_WHEEL;

    canmsg.data[0] = speedR;
    canmsg.data[1] = 0;  
    canmsg.data[2] = dirR;
    canmsg.data[3] = aibotgo; 
    canmsg.data[4] = speedL;
    canmsg.data[5] = 0;  
    canmsg.data[6] = dirL;
    canmsg.data[7] = aibotgo; 	
  //ROS_INFO("[L]speedCmd= %x dir= %x [R]speedCmd= %x dir= %x \n", canmsg.data[0],
      //canmsg.data[2],canmsg.data[4], canmsg.data[6]);  
	  
	printf("canmsg:%d.%d.%d.%d ", canmsg.data[0],canmsg.data[1],canmsg.data[2],canmsg.data[3]);
	printf("  %d.%d.%d.%d\n", canmsg.data[4],canmsg.data[5],canmsg.data[6],canmsg.data[7]);

#if 1
    if (send_port(&canmsg)) {
        //printf("message sent !!\n");
    } else {
        printf("send fail L\n");
    } 
#endif 
    return 0;   
}



