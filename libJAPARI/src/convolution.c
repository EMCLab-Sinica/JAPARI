
#include "convolution.h"

void __JAP_FETCH_INPUT_CONV(JAP_DATA *D_IN, _q15* DST , JAP_INTER_IDX* itr, JAP_INTRA_IDX* ita, JAP_TILE_SIZE* ts, uint16_t batch, uint8_t pg, uint8_t sign);
void __JAP_FETCH_WEIGHT(JAP_PARA *PARA, JAP_INTER_IDX* itr, JAP_INTRA_IDX* ita, JAP_TILE_SIZE* ts, uint16_t batch ,int dummy_w, int loop_cnt );
void __mpypara_init(msp_matrix_mpy_q15_params* mpyParams, JAP_TILE_SIZE* ts, uint16_t batch);
msp_status __JAP_matrix_mpy_q15(const msp_matrix_mpy_q15_params *params, const _q15 *srcA, const _q15 *srcB, _q15 *dst, uint32_t destNV , uint16_t len);
void __JAP_ADD(const msp_add_q15_params *params, const _q15 *srcA, const _q15 *srcB, _q15 *dst ,uint32_t destNV);


void __JAP_PG_RETRIEVE(JAP_LAYER* LAYER,  JAP_INTER_IDX* itr, JAP_INTRA_IDX* ita, JAP_TILE_SIZE* ts, int* l_cnt);

void __JAP_PG_RETRIEVE(JAP_LAYER* LAYER,  JAP_INTER_IDX* itr, JAP_INTRA_IDX* ita, JAP_TILE_SIZE* ts, int* l_cnt){
	SPI_ADDR ADDR;
	JAP_DATA *D_IN    = &LAYER->DATA_IN;
	JAP_DATA *D_OUT   = &LAYER->DATA_OUT;
	JAP_PARA *PARA    = &LAYER->PARA;
	uint16_t KC   = PARA->KERNEL_W;
	uint16_t KR   = PARA->KERNEL_H;
	uint16_t ROWS = D_OUT->H;
	uint16_t COLS = D_OUT->W;
	uint16_t MCH  = D_OUT->CH;
	uint16_t NCH  = D_IN->CH;

	uint32_t D_IN_Ptr   = D_IN->DATA_Ptr;
	uint32_t D_OUT_Ptr  = D_OUT->DATA_Ptr;
	uint32_t WEIGHT_Ptr = PARA->WEIGHT;
	uint32_t PB         = LAYER->BUFFER_Ptr;

	uint16_t batch = LAYER->BATCH;
	uint16_t NFPs = 0; //uum_fp
	uint16_t prog_oft = 0;
	uint16_t trans=0;
	uint16_t MCH_F =  MCH+(MCH + batch-1)/batch;
	uint16_t mch_p = 0;
	uint32_t b_offset = Aoffset3D( ROWS , COLS , MCH_F ,COLS, MCH_F)*sizeof(_q15);

	_q15* srcA;
	_q15* srcB;
	uint32_t srcC32;

	int pb_1,pb_L;
	int LL_MARKER=0;

	uint16_t lea_w_offset = Tc * Tr * (Tn+2);
		lea_w_offset += (lea_w_offset & 0x1);
	uint16_t lea_o_offset = lea_w_offset + (Tn+2) * Tm*2;
		lea_o_offset += (lea_o_offset &0x1);


	ADDR.L = PB + sizeof(_q15)*batch;
	SPI_READ(&ADDR,(uint8_t*)&pb_1,sizeof(_q15));
	if( (  (pb_1 <= 0) && (LAYER->SIGN==0)) || ((pb_1 >= 0) && (LAYER->SIGN!=0))  ){
		return;
	}else{
		*l_cnt = (LAYER->SIGN==0) ? pb_1 : pb_1 * -1;
		uint16_t tmp=1;
		for(itr->r=0; itr->r<ROWS ; itr->r+=Tr){for(itr->c=0; itr->c<COLS ; itr->c+=Tc){
			for(itr->m=0; itr->m <MCH  ; itr->m +=Tm){for(itr->n=0;  itr->n <NCH  ; itr->n +=Tn){
				for(itr->kr=0; itr->kr < KR ; itr->kr++ ){for(itr->kc=0; itr->kc < KC ; itr->kc++ ){
					if(tmp == *l_cnt){
						ts->tr = (itr->r+Tr > ROWS) ? (ROWS - itr->r) : Tr;
						ts->tc = (itr->c+Tc > ROWS) ? (COLS - itr->c) : Tc;
						ts->tm = (itr->m+Tm > ROWS) ? (MCH  - itr->m) : Tm;
						ts->tn = (itr->n+Tn > ROWS) ? (NCH  - itr->n) : Tn;
						NFPs = ( ts->tm + batch-1) / batch ;

						srcA = LEA_MEMORY+lea_o_offset;

						ADDR.L = PB + Aoffset3D(ts->tr-1,ts->tc-1, (ts->tm + NFPs) - 1 , ts->tc , ts->tm + NFPs ) * sizeof(_q15);
						if( ts->tm < batch){
							ADDR.L  +=  ((batch - ts->tm)*sizeof(_q15));
						}
						SPI_READ(&ADDR,(uint8_t*)&pb_L, sizeof(_q15));

						if( pb_1 != pb_L ){
							ita->op = 0;
							ADDR.L = PB ;
							if( ts->tm < batch){
								ADDR.L  +=  ((batch - ts->tm)*sizeof(_q15));
							}
							for(ita->r=0;ita->r < ts->tr;ita->r++){
								for(ita->c=0;ita->c < ts->tc;ita->c++){
									SPI_READ(&ADDR,(uint8_t*)srcA, ts->tm + NFPs);
									if(srcA[batch] != srcA[ts->tm + NFPs -1])return;
									srcA += ((ts->tm + NFPs) + ((ts->tm + NFPs)&0x01))  ;
									ADDR.L += ((ts->tm + NFPs)*sizeof(_q15));
								}
							}
							*ita = (JAP_INTRA_IDX ){.r=0, .c=0, .m=0, .n=0, .op=1, .flip=0};
							return;
						}else{
							ita->op = 1;
							uint16_t sz = MCH + (MCH +batch-1 / batch);

							for(ita->r = 0; ita->r <  ts->tr  ; ita->r++ ){
								ADDR.L = (ita->flip^0x1)*b_offset + D_OUT_Ptr + Aoffset3D( itr->r+ ita->r, itr->c + ita->c , (itr->m+ ita->m)+ batch + ((itr->m+ ita->m)/batch),COLS, MCH_F)*sizeof(_q15);

								SPI_READ(&ADDR,(uint8_t*)&pb_L,sizeof(_q15));
								if(LAYER->SIGN==0){if(pb_L >= LL_MARKER){LL_MARKER = pb_L;}else{break;}}
								else{if(pb_L <= LL_MARKER){LL_MARKER = pb_L;}else{break;}}
								if(ita->r == (ts->tr-1) )break;
							}
							LL_MARKER=0;
							for(ita->c = 0; ita->c <  ts->tc  ; ita->c++ ){
								ADDR.L = (ita->flip^0x1)*b_offset + D_OUT_Ptr + Aoffset3D( itr->r+ ita->r, itr->c + ita->c , (itr->m+ ita->m)+ batch + ((itr->m+ ita->m)/batch),COLS, MCH_F)*sizeof(_q15);
								SPI_READ(&ADDR,(uint8_t*)&pb_L,sizeof(_q15));
								if(LAYER->SIGN==0){if(pb_L >= LL_MARKER){LL_MARKER = pb_L;}else{break;}}
								else{if(pb_L <= LL_MARKER){LL_MARKER = pb_L;}else{break;}}
								if(ita->c == (ts->tc-1) )break;
							}
							LL_MARKER=0;
							for(ita->m = 0 ; ita->m < ts->tm ; ita->m+=batch ){
								ADDR.L = (ita->flip^0x1)*b_offset + D_OUT_Ptr + Aoffset3D( itr->r+ ita->r, itr->c + ita->c , (itr->m+ ita->m)+ batch + ((itr->m+ ita->m)/batch),COLS, MCH_F)*sizeof(_q15);
								SPI_READ(&ADDR,(uint8_t*)&pb_L,sizeof(_q15));
								if(LAYER->SIGN==0){if(pb_L >= LL_MARKER){LL_MARKER = pb_L;}else{return;}}
								else{if(pb_L <= LL_MARKER){LL_MARKER = pb_L;}else{return;}}
								if(ita->m >= (ts->tm-1) )ita->m= ts->tm -1; return;
							}
						}
					}else{tmp++;ita->flip^=0x1;}
				}}}ita->flip=0;}}}
	}
	*itr = (JAP_INTER_IDX) {.r=0, .c=0, .m=0, .n=0, .kr=0, .kc=0} ;
	*ita = (JAP_INTRA_IDX ){.r=0, .c=0, .m=0, .n=0, .op=0, .flip=0};
	*l_cnt=1;


}
msp_status __JAP_matrix_mpy_q15(const msp_matrix_mpy_q15_params *params, const _q15 *srcA, const _q15 *srcB, _q15 *dst, uint32_t destNV , uint16_t len)
{
	SPI_ADDR ADDR;
    uint16_t srcARows;
    uint16_t srcACols;
    uint16_t srcBRows;
    uint16_t srcBCols;
    msp_status status;
    MSP_LEA_MPYMATRIXROW_PARAMS *leaParams;
    _q15 *dstTX = dst;

    /* Initialize the row and column sizes. */
    srcARows = params->srcARows;
    srcACols = params->srcACols;
    srcBRows = params->srcBRows;
    srcBCols = params->srcBCols;
    if(srcBCols == 0)return 0;

    /* Allocate MSP_LEA_MPYMATRIXROW_PARAMS structure. */
    leaParams = (MSP_LEA_MPYMATRIXROW_PARAMS *)msp_lea_allocMemory(sizeof(MSP_LEA_MPYMATRIXROW_PARAMS)/sizeof(uint32_t));

    /* Set status flag. */
    status = MSP_SUCCESS;


	/* Set MSP_LEA_MPYMATRIXROW_PARAMS structure. */
	leaParams->rowSize = srcBRows;
	leaParams->colSize = srcBCols;
	leaParams->colVector = MSP_LEA_CONVERT_ADDRESS(srcB);
	leaParams->output = MSP_LEA_CONVERT_ADDRESS(dst);

	/* Load source arguments to LEA. */
	LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(srcA);
	LEAPMS1 = MSP_LEA_CONVERT_ADDRESS(leaParams);

	/* Invoke the LEACMD__MPYMATRIXROW command. */
	LEAPMCB =  LEACMD__MPYMATRIXROW | LEAITFLG1;

	ADDR.L = destNV;
	SPI_WRITE(&ADDR,(uint8_t*)dstTX, len * sizeof(_q15));

	while(LEACNF1 & LEABUSY__BUSY);


    /* Free MSP_LEA_MPYMATRIXROW_PARAMS structure. */
    msp_lea_freeMemory(sizeof(MSP_LEA_MPYMATRIXROW_PARAMS)/sizeof(uint32_t));

    return status;
}

void __JAP_ADD(const msp_add_q15_params *params, const _q15 *srcA, const _q15 *srcB, _q15 *dst ,uint32_t destNV)
{
	SPI_ADDR ADDR;
    uint16_t length;
    msp_status status;
    MSP_LEA_ADDMATRIX_PARAMS *leaParams;
    _q15 *dstTX = dst;

    /* Initialize the vector length. */
    length = params->length;
    if (length==0)return 0;

    /* Allocate MSP_LEA_ADDMATRIX_PARAMS structure. */
    leaParams = (MSP_LEA_ADDMATRIX_PARAMS *)msp_lea_allocMemory(sizeof(MSP_LEA_ADDMATRIX_PARAMS)/sizeof(uint32_t));

    /* Set MSP_LEA_ADDMATRIX_PARAMS structure. */
    leaParams->input2 = MSP_LEA_CONVERT_ADDRESS(srcB);
    leaParams->output = MSP_LEA_CONVERT_ADDRESS(dst);
    leaParams->vectorSize = length;
    leaParams->input1Offset = 1;
    leaParams->input2Offset = 1;
    leaParams->outputOffset = 1;

    /* Load source arguments to LEA. */
    LEAPMS0 = MSP_LEA_CONVERT_ADDRESS(srcA);
    LEAPMS1 = MSP_LEA_CONVERT_ADDRESS(leaParams);

    /* Invoke the LEACMD__ADDMATRIX command. */
    LEAPMCB =  LEACMD__ADDMATRIX | LEAITFLG1;
    ADDR.L = destNV;
    SPI_WRITE(&ADDR,(uint8_t*)dstTX,sizeof(_q15) * length);

    while(LEACNF1 & LEABUSY__BUSY);
    /* Free MSP_LEA_ADDMATRIX_PARAMS structure. */
    msp_lea_freeMemory(sizeof(MSP_LEA_ADDMATRIX_PARAMS)/sizeof(uint32_t));
}


void __JAP_FETCH_INPUT_CONV(JAP_DATA *D_IN, _q15* DST , JAP_INTER_IDX* itr, JAP_INTRA_IDX* ita, JAP_TILE_SIZE* ts, uint16_t batch, uint8_t pg, uint8_t sign){
	uint16_t dummy_i = sign == 1 ? 0x8000 : 0x0800 ;
	SPI_ADDR ADDR;
	if(pg == 0){
		ADDR.L = D_IN->DATA_Ptr + Aoffset3D( itr->r  + ita->r + itr->kr ,itr->c + ita->c + itr->kc , itr->n, D_IN->W ,D_IN->CH) * sizeof(_q15);
		if(ts->tn & 0x1){
			SPI_READ(&ADDR,(uint8_t *)DST,ts->tn*sizeof(_q15));
			DST[ts->tn] = dummy_i;
		}else{
			SPI_READ(&ADDR,(uint8_t *)DST,ts->tn*sizeof(_q15));
			DST[ts->tn] = dummy_i;
			DST[ts->tn+1] = 0;
		}
	}
	else{
		uint16_t NFPs = 0;
			NFPs = ( ts->tn + batch-1) / batch ;
		uint16_t nps = 0;
			nps = ( itr->n ) / batch ;

		ADDR.L = D_IN->DATA_Ptr + Aoffset3D( itr->r  + ita->r + itr->kr , itr->c + ita->c + itr->kc , itr->n + nps , D_IN->W , itr->n + NFPs) * sizeof(_q15);
		_q15* dst = DST;
		uint16_t length_n  = ts->tn +NFPs;
		uint16_t length_n2 = ts->tn;
		uint16_t cur = 0;
		uint16_t trans = 0;
		SPI_READ(&ADDR,(uint8_t*)SRAM_BUFFER,length_n * sizeof(_q15));

		for(uint16_t n = 0; n < NFPs ; n++){
			uint16_t bound = (cur + batch) < ts->tn ? cur + batch :ts->tn;
			for(cur ; cur < bound; cur++){
				dst[cur] = SRAM_BUFFER[cur+n];
			}
		}
		if(ts->tn & 0x1){
			dst[ts->tn] = dummy_i;
		}else{
			dst[ts->tn] = dummy_i;
			dst[ts->tn+1] = 0;
		}
	}
}


void __JAP_FETCH_WEIGHT(JAP_PARA *PARA , JAP_INTER_IDX* itr, JAP_INTRA_IDX* ita, JAP_TILE_SIZE* ts, uint16_t batch ,int dummy_w, int loop_cnt){
	SPI_ADDR ADDR;
	uint32_t W_Ptr= PARA->WEIGHT;
	uint16_t MCH  = PARA->CH_OUT;
	uint16_t NCH  = PARA->CH_IN;
	uint16_t KC   = PARA->KERNEL_W;

	uint16_t NFPs = 0;
		NFPs = ( ts->tm + batch-1) / batch ;
	uint16_t lea_w_offset = Tc * Tr * (Tn+2);
		lea_w_offset += (lea_w_offset & 0x1);
	uint16_t lea_o_offset = lea_w_offset + (Tn+2) * Tm*2;
		lea_o_offset += (lea_o_offset &0x1);

	ADDR.L =W_Ptr + Aoffset4D(itr->kr,itr->kc,itr->n,itr->m,KC,NCH,MCH)*sizeof(_q15);
	_q15* srcW  = LEA_MEMORY + lea_w_offset;

	int fps = NFPs-1;
	uint16_t nws = ts->tm;
	uint16_t step= ts->tm % batch;

	uint16_t length = ts->tn;
	while(length--){
		SPI_READ(&ADDR,(uint8_t*)srcW,ts->tm*sizeof(_q15));
		if( batch >= ts->tm ){srcW[ts->tm] = 0;srcW[ts->tm+1] = 0;}
		else{
			fps = NFPs-1;
			nws = ts->tm;
			step= ts->tm % batch;

			for(fps;fps>=0;fps--){
				srcW[nws+fps] = 0;
				for(uint16_t cur=(nws-batch+step); cur < nws ; cur++){
					srcW[fps + cur] = srcW[cur];
				}
				nws -= (batch - step ) ;
				step = 0;
			}
			if( (NFPs + ts->tm) & 0x1)srcW[NFPs + ts->tm]=0;
		}
		ADDR.L += MCH*sizeof(_q15);
		srcW += ( ts->tm + NFPs + ((ts->tm + NFPs)&0x1)  );
	}

	fps = NFPs-1;
	nws = ts->tm;
	step= ts->tm % batch;

	for(fps;fps>=0;fps--){
		srcW[nws+fps] = dummy_w * loop_cnt;
		for(uint16_t cur=(nws-batch+step); cur < nws ; cur++){
			srcW[fps + cur] = 0;
		}
		nws -= (batch - step ) ;
		step = 0;
	}

	if( (ts->tn & 0x1) == 0 ){
		srcW += ( ts->tm + NFPs + ((ts->tm + NFPs)&0x1)  );
		for(uint16_t i=0;i < ( ts->tm + NFPs + ((ts->tm + NFPs)&0x1)  );i++)srcW[i]=0;
	}
}

void __mpypara_init(msp_matrix_mpy_q15_params* mpyParams, JAP_TILE_SIZE* ts, uint16_t batch){
	uint16_t Cols = (( ts->tm + batch-1) / batch ) + ts->tm;
	mpyParams->srcARows = 1;
	mpyParams->srcBCols = Cols + (Cols &0x1) ;
	if(ts->tn & 0x1){
		mpyParams->srcACols = ts->tn+1;
		mpyParams->srcBRows = ts->tn+1;
	}
	else{
		mpyParams->srcACols = ts->tn+2;
		mpyParams->srcBRows = ts->tn+2;
	}
}

void JAP_CONV(JAP_LAYER* LAYER){
	SPI_ADDR ADDR;
	JAP_DATA *D_IN    = &LAYER->DATA_IN;
	JAP_DATA *D_OUT   = &LAYER->DATA_OUT;
	JAP_PARA *PARA    = &LAYER->PARA;
	uint16_t KC   = PARA->KERNEL_W;
	uint16_t KR   = PARA->KERNEL_H;
	uint16_t ROWS = D_OUT->H;
	uint16_t COLS = D_OUT->W;
	uint16_t MCH  = D_OUT->CH;
	uint16_t NCH  = D_IN->CH;

	uint32_t D_IN_Ptr   = D_IN->DATA_Ptr;
	uint32_t D_OUT_Ptr  = D_OUT->DATA_Ptr;
	uint32_t WEIGHT_Ptr = PARA->WEIGHT;
	uint32_t PB         = LAYER->BUFFER_Ptr;

	uint16_t batch = LAYER->BATCH;
	uint16_t NFPs = 0; //uum_fp
	uint16_t prog_oft = 0;
	uint16_t trans=0;
	uint16_t MCH_F =  MCH+(MCH + batch-1)/batch;
	uint16_t mch_p = 0;
	uint32_t b_offset = Aoffset3D( ROWS , COLS , MCH_F ,COLS, MCH_F)*sizeof(_q15);

	int dummy_w =LAYER->SIGN ? 0x0001 : 0x0010;
	int loop_cnt = 1;
	int LL_MARKER=0;
	int pb_1=1;
	int pb_L=1;


	msp_add_q15_params addParams;
	msp_matrix_mpy_q15_params mpyParams;
	_q15* srcA;
	_q15* srcB;
	uint32_t srcC32;

	uint16_t lea_w_offset = Tc * Tr * (Tn+2);
		lea_w_offset += (lea_w_offset & 0x1);
	uint16_t lea_o_offset = lea_w_offset + (Tn+2) * Tm*2;
		lea_o_offset += (lea_o_offset &0x1);

	JAP_INTER_IDX itr = {.r=0, .c=0, .m=0, .n=0, .kr=0, .kc=0} ;
	JAP_INTRA_IDX ita = {.r=0, .c=0, .m=0, .n=0, .op=0, .flip=0};
	JAP_TILE_SIZE ts  = {.tr=Tr, .tc=Tc, .tm=Tm, .tn=Tn};

	//progress search
	__JAP_PG_RETRIEVE(LAYER,  &itr, &ita, &ts, &loop_cnt);
	//computing
	for(; itr.r<ROWS ; itr.r+=Tr){for(; itr.c<COLS ; itr.c+=Tc){
		for(; itr.m <MCH  ; itr.m +=Tm){for(;  itr.n <NCH  ; itr.n +=Tn){
			ts.tr = (itr.r+Tr > ROWS) ? (ROWS - itr.r) : Tr;
			ts.tc = (itr.c+Tc > ROWS) ? (COLS - itr.c) : Tc;
			ts.tm = (itr.m+Tm > ROWS) ? (MCH  - itr.m) : Tm;
			ts.tn = (itr.n+Tn > ROWS) ? (NCH  - itr.n) : Tn;

			NFPs = ( ts.tm + batch-1) / batch ;
			__mpypara_init(&mpyParams, &ts,batch);
			//intra
			for(; itr.kr < KR ; itr.kr++ ){for(; itr.kc < KC ; itr.kc++ ){
				for(; ita.op<2 ;ita.op++){
					if(ita.op == 0){
						__JAP_FETCH_WEIGHT(PARA,&itr,&ita,&ts,batch,dummy_w,loop_cnt);
						srcA = LEA_MEMORY+lea_w_offset;
						srcB = LEA_MEMORY+lea_o_offset;

						srcC32 = PB + (ita.r * ts.tc + ita.c) * ( (ts.tm + NFPs) + ((ts.tm + NFPs)&0x01) );
						if( ts.tm < batch){
							srcC32 +=  ((batch - ts.tm)*sizeof(_q15));
						}
						for(;ita.r < ts.tr;ita.r++){
							for(;ita.c < ts.tc;ita.c++){
								__JAP_FETCH_INPUT_CONV(D_IN, LEA_MEMORY , &itr, &ita, &ts, batch, D_IN->PG, LAYER->SIGN);
								__JAP_matrix_mpy_q15(&mpyParams,LEA_MEMORY,srcA,srcB,srcC32,(ts.tm + NFPs));
								srcB += ((ts.tm + NFPs) + ((ts.tm + NFPs)&0x01))  ;
								srcC32 += ((ts.tm + NFPs)*sizeof(_q15));
							}ita.c=0;//end loop sub_op_r
						}ita.r=0;//end loop sub_op_r
					}else{
						srcA = LEA_MEMORY+lea_o_offset;
						if( ts.tm < batch){
							srcA += ( batch - ts.tm);
						}
						prog_oft =  Aoffset3D(ita.r ,ita.c , ita.m + ita.m/batch,ts.tc, (ts.tm + NFPs) + ((ts.tm + NFPs)&0x01) );
						srcA += prog_oft;

						for(;ita.r < ts.tr;ita.r++){
							for(;ita.c < ts.tc;ita.c++){
								if( (itr.kr==0) &&  (itr.kc==0) && (itr.n==0)){
									ADDR.L = D_OUT_Ptr + Aoffset3D( itr.r+ ita.r, itr.c + ita.c , (itr.m+ ita.m) + ((itr.m+ ita.m)/batch),COLS, MCH_F)*sizeof(_q15);
									trans = ts.tm + NFPs - (ita.m + ita.m/batch );
									SPI_WRITE(&ADDR,(uint8_t*)srcA,trans*sizeof(_q15));
									srcA += (trans  + ((ts.tm + NFPs)&0x01) ) ;
								}else{
									//prev
									ADDR.L = (ita.flip^0x1)*b_offset + D_OUT_Ptr + Aoffset3D( itr.r+ ita.r, itr.c + ita.c , (itr.m+ ita.m) + ((itr.m+ ita.m)/batch),COLS, MCH_F)*sizeof(_q15);
									trans = ts.tm + NFPs - (ita.m + ita.m/batch );

									SPI_READ(&ADDR,(uint8_t*)LEA_MEMORY,trans*sizeof(_q15));

									addParams.length = trans + (trans&0x1);
									ADDR.L = (ita.flip)*b_offset + D_OUT_Ptr + Aoffset3D( itr.r+ ita.r, itr.c + ita.c , (itr.m+ ita.m) + ((itr.m+ ita.m)/batch),COLS, MCH_F)*sizeof(_q15);
									__JAP_ADD(&addParams, srcA, LEA_MEMORY, LEA_MEMORY,ADDR.L);
									srcA += (trans  + ((ts.tm + NFPs)&0x01) ) ;

								}
								ita.m = 0;

							}ita.c=0;//end loop sub_op_r
						}ita.r=0;//end loop sub_op_r


					}
				}ita.op=0;loop_cnt++;ita.flip^=0x1;
			}itr.kc;}itr.kr=0;
		}itr.n=0;ita.flip=0;}itr.m=0;
	}itr.c=0;}itr.r=0;
	LAYER->SIGN = LAYER->SIGN ? 0 : 1 ;
}

