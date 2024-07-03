////////////////////////////////////
//    jetsonai@jetsonai.co.kr
//          20230604
///////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#include <cmath>

#include "cansocket.hpp"

int soc;
int read_can_port;
struct sockaddr_can addr;

int open_port(const char *port)
{
    struct ifreq ifr;


    /* open socket */
    soc = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if(soc < 0)
    {
        return (-1);
    }

  strcpy(ifr.ifr_name, port );
  ioctl(soc, SIOCGIFINDEX, &ifr);

  memset(&addr, 0, sizeof(addr));
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(soc, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    {

        return (-1);
    }

    return 0;
}

int send_port(struct can_frame *frame_s)
{
    int nbytes;
    frame_s->can_dlc = 8;
    /*
     printf(">> frame_s->data[0] 0x%x\n", frame_s->data[0]);
       printf(">> frame_s->data[1] 0x%x\n", frame_s->data[1]);
       printf(">> frame_s->data[2] 0x%x\n", frame_s->data[2]);
       printf(">> frame_s->data[3] 0x%x\n", frame_s->data[3]);
       printf(">> frame_s->data[4] 0x%x\n", frame_s->data[4]);
       printf(">> frame_s->data[5] 0x%x\n", frame_s->data[5]);
       printf(">> frame_s->data[6] 0x%x\n", frame_s->data[6]);
       printf(">> frame_s->data[7] 0x%x\n", frame_s->data[7]);
//*/
  nbytes = write(soc, frame_s, sizeof(struct can_frame));
   
  return nbytes;
}

 
int read_port(struct can_frame *frame_rd)
{
    int nbytes = 0;
    nbytes = read(soc, frame_rd, sizeof(struct can_frame));

    if (nbytes < 0) {
        perror("can raw socket read");
        return -1;
    }

    /* paranoid check ... */
    if (nbytes < sizeof(struct can_frame)) {
            fprintf(stderr, "read: incomplete CAN frame\n");
        return -1;
    }
    
  return nbytes;

}

int close_port()
{
    close(soc);
    return 0;
}

/*-------------------------------------------------*/
void convert10(char* tempv, uint32_t *decnum) 
{
    int lenv = 8;
    int pos = 0;
    uint32_t u32=0;

    for(int i = lenv-1; i>=0;i--) {
        int num1 = tempv[i];
        if(num1 >=48 && num1 <=57) { //0~9
            u32 += (num1-48) * pow(16, pos);
        } else if(num1 >=65 && num1 <=70) { //A~F
            u32 += (num1-(65-10)) * pow(16, pos);
        } else if(num1 >=97 && num1 <=102) { //a~f
            u32 += (num1-(97-10)) * pow(16, pos);
        }    
        pos++;
    }  
    *decnum = u32;
}

//------------------------------------------------------------------------------
uint32_t interprete32(uint8_t * msgdata)
{
    char tempv[32+1];
    uint32_t u32 = 0;
	       /*printf("interprete32>> msgdata[0] 0x%x\n", msgdata[0]);
	       printf("interprete32>> msgdata[1] 0x%x\n", msgdata[1]);
	       printf("interprete32>> msgdata[2] 0x%x\n", msgdata[2]);
	       printf("interprete32>> msgdata[3] 0x%x\n", msgdata[3]);*/
    sprintf(tempv,"%02x%02x%02x%02x", msgdata[0], msgdata[1], msgdata[2], msgdata[3]);
    int lenv = strlen(tempv);
    tempv[lenv] = 0;

    convert10(tempv, &u32);
    //printf("tempv:%s <interprete32> : %d\n", tempv, u32);
    return u32;
}


#if 0
int main(void)
{
    open_port("can0");
    read_port();
    return 0;
}
#endif
