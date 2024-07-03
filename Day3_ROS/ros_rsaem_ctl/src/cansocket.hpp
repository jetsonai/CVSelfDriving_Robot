////////////////////////////////////
//    jetsonai@jetsonai.co.kr
//          20230604
///////////////////////////////////
#ifndef CANSOCKET_H
#define CANSOCKET_H

#include <linux/can.h>
#include <linux/can/raw.h>
#include <stdint.h>

#if 0
/**
 * struct can_frame - basic CAN frame structure
 * @can_id:  CAN ID of the frame and CAN_*_FLAG flags, see canid_t definition
 * @can_dlc: frame payload length in byte (0 .. 8) aka data length code
 *           N.B. the DLC field from ISO 11898-1 Chapter 8.4.2.3 has a 1:1
 *           mapping of the 'data length code' to the real payload length
 * @__pad:   padding
 * @__res0:  reserved / padding
 * @__res1:  reserved / padding
 * @data:    CAN frame payload (up to 8 byte)
 */
struct can_frame {
	canid_t can_id;  /* 32 bit CAN_ID + EFF/RTR/ERR flags */
	__u8    can_dlc; /* frame payload length in byte (0 .. CAN_MAX_DLEN) */
	__u8    __pad;   /* padding */
	__u8    __res0;  /* reserved / padding */
	__u8    __res1;  /* reserved / padding */
	__u8    data[CAN_MAX_DLEN] __attribute__((aligned(8)));
};

#endif


int open_port(const char *port);
int send_port(struct can_frame *frame);
int read_port(struct can_frame *frame_rd);
int close_port();
uint32_t interprete32(uint8_t * msgdata);
void convert10(char* tempv, uint32_t *decnum) ;



#endif
