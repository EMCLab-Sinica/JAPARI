#include "japari.h"


DSPLIB_DATA(LEA_MEMORY,4)
_q15 LEA_MEMORY[2048-LEA_STACK];

_q15 SRAM_BUFFER[3*Tm];

inline uint32_t Aoffset2D(uint16_t V2,uint16_t V1,uint16_t D1){
	return (uint32_t) (uint32_t)(V2 * D1) + (uint32_t)(V1);
}

inline uint32_t Aoffset3D(uint16_t V3,uint16_t V2,uint16_t V1,uint16_t D2,uint16_t D1){
	return (uint32_t)(V3 * D2 * D1) + (uint32_t)(V2 * D1) + (uint32_t)(V1);
}
inline uint32_t Aoffset4D(uint16_t V4,uint16_t V3,uint16_t V2,uint16_t V1,uint16_t D3,uint16_t D2,uint16_t D1){
	return (uint32_t)(V4 * D3 * D2 * D1) + (uint32_t)(V3 * D2 * D1) + (uint32_t)(V2 * D1) + (uint32_t)(V1);
}

void JAP_INFERENCE(JAP_NETWORK *net){

	int i;
	for(i = net->FOOTPRINT ; i < net->TOTAL_LAYERS ; i++){
		JAP_LAYER *LAYER = &(net->LAYERS[net->FOOTPRINT]);
		JAP_LAYER *NEXT_LAYER = &net->LAYERS[(net->FOOTPRINT+1) % (net->TOTAL_LAYERS)];
		LAYER->fun(LAYER);
		net->FOOTPRINT++;
	}
	net->FOOTPRINT=0;
}
