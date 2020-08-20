#ifndef JAPARILIB_JAPARI_H_
#define JAPARILIB_JAPARI_H_

#include <math.h>
#include <stdint.h>
#include <stdbool.h>

#include "DSPLib.h"
#include "driverlib.h"
#include "extfram.h"

#define SIZE_of_Q15 2
#define Tr 4
#define Tc 4
#define Tn 8
#define Tm 8

#define LEA_STACK 200

extern uint16_t UART_FG;
extern _q15 LEA_MEMORY[2048-LEA_STACK];

//SRAM BUFFER
extern _q15 SRAM_BUFFER[3*Tm];

typedef struct{
	uint32_t WEIGHT;
	uint16_t KERNEL_W;
	uint16_t KERNEL_H;
	uint16_t CH_IN;
	uint16_t CH_OUT;
	uint32_t BIAS;
}JAP_PARA;

typedef struct{
	uint32_t DATA_Ptr;
	uint16_t CH;
	uint16_t W;
	uint16_t H;
	uint8_t PG;
}JAP_DATA;

typedef struct HAWL{
	void (*fun)();
	JAP_DATA DATA_IN;
	JAP_DATA DATA_OUT;
	JAP_DATA PARA;
	uint8_t  SIGN;
	uint32_t BUFFER_Ptr;
	uint32_t BATCH;
}JAP_LAYER;

typedef struct{
	JAP_LAYER* LAYERS;
	volatile uint16_t FOOTPRINT;
	uint16_t TOTAL_LAYERS;
}JAP_NETWORK;

typedef struct{
	uint32_t PTR;
	uint16_t ROWS;
	uint16_t COLS;
	uint16_t MCH;
	uint16_t NCH;
	uint16_t KR;
	uint16_t KC;
	uint16_t _row;
	uint16_t _col;
	uint16_t _m;
	uint16_t _n;
	uint16_t _kr;
	uint16_t _kc;
	uint16_t _row_step;
	uint16_t _col_step;
	uint16_t _m_step;
	uint16_t _n_step;
	uint16_t P_flag;
	uint16_t _c_offset;
	uint16_t _r_offset;
}JAP_TILE;

#include "myuart.h"
#include "pool.h"
#include "convolution.h"
#include "nonlinear.h"

uint32_t Aoffset2D(uint16_t V2,uint16_t V1,uint16_t D1);
uint32_t Aoffset3D(uint16_t V3,uint16_t V2,uint16_t V1,uint16_t D2,uint16_t D1);
uint32_t Aoffset4D(uint16_t V4,uint16_t V3,uint16_t V2,uint16_t V1,uint16_t D3,uint16_t D2,uint16_t D1);
void JAP_INFERENCE(JAP_NETWORK *net);


#endif /* JAPARILIB_JAPARI_H_ */
