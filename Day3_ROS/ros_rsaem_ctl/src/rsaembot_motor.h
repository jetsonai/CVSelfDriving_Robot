#ifndef AIBOT_CORE_CONFIG_H_
#define AIBOT_CORE_CONFIG_H_


#include <math.h>

//#define IMU_FUNCTION

//#define TICK2RAD                         0.001533981  // 0.087890625[deg] * 3.14159265359 / 180 = 0.001533981f

/*-------------aibot----------------*/
#include <pthread.h>
#include <signal.h>
#include "cansocket.hpp"

/*-------------aibot----------------*/

#define CONTROL_MOTOR_SPEED_FREQUENCY          30   //hz
#define CONTROL_MOTOR_TIMEOUT                  500  //ms
#define IMU_PUBLISH_FREQUENCY                  200  //hz
#define CMD_VEL_PUBLISH_FREQUENCY              30   //hz
#define DRIVE_INFORMATION_PUBLISH_FREQUENCY    30   //hz


#define WHEEL_NUM                        2

#define LEFT                             0
#define RIGHT                            1

#define LINEAR                           0
#define ANGULAR                          1

#define DEG2RAD(x)                       (x * 0.01745329252)  // *PI/180
#define RAD2DEG(x)                       (x * 57.2957795131)  // *180/PI

#define TICK2RAD                         0.001498851  // 0.085877863[deg] * 3.14159265359 / 180 = 0.001533981f


uint32_t current_offset;



/*******************************************************************************
* aibot
*******************************************************************************/

#define AIBOT_GO 0
#define AIBOT_STOP 1

#define MSG_AIBOT_WHEEL 0x01
#define MSG_AIBOT_ENCODER 0x11
#define MSG_AIBOT_ENC_RESET 0x09

#define PI_M 3.141592654

#define CAN_ENC_OFFSET 2147483647

#define lengthBetweenTwoWheels 0.21
#define wheelRadius 0.004

//#define RPR_M (131*16*2)
#define RPR_M (100*13*2)

//#TICK4RAD ((360.0/RPR_M)*(180.0/PI_M))
//#define RPR_M (131*16*2)

#define DistancePerCount ((PI_M * wheelRadius * 2) / RPR_M) // (2*PI*r)/ppr

int32_t PrevLeftEncoder;
int32_t PrevRightEncoder;
int dirL, dirR;

int sendCanTx(int speed, int dir, int heading, int aibotgo);
void *recvThread(void* data);

float motorLeft, motorRight;

float steering_gain_value, steering_dgain_value;


int RunningMode_F;
int RunningMode_B;

int SaftyMode_F;
int SaftyMode_B;


#endif // AIBOT_CORE_CONFIG_H_

