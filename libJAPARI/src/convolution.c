
#include "convolution.h"

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

	uint16_t _tr = 0;
	uint16_t _tc = 0;
	uint16_t _tm = 0;
	uint16_t _tn = 0;
	uint16_t OP_L= 0;

	int PB_1=1;
	int PB_L=1;


	int dummy_w =LAYER->SIGN ? 0x0001 : 0x0010;
	int loop_cnt = 0;

	uint16_t _row, _col, _n, _m, _kc, _kr;
	uint16_t sub_op_r=0,sub_op_c=0,sub_op_m=0;
	int LL_MARKER=0;

	uint16_t rrr,ccc,mmm,nnn,kkr,kkc,temp;temp=0;
	msp_add_q15_params addParams;
	uint16_t lea_w_offset = Tc * Tr * (Tn+2);
	uint16_t lea_o_offset = Tc * Tr * (Tn+2) + (Tn+2) * Tm*2;

	_q15 * temp_2;
	msp_matrix_mpy_q15_params mpyParams;
	JAP_TILE tile_i ;


	_q15* srcA;
	_q15* srcB;

	uint16_t length ;

	uint16_t offset_prog ;
	uint16_t length_r ;
	uint16_t length_c ;

	ADDR.L = PB+2;
	SPI_READ(&ADDR,(uint8_t*)&PB_1,2);
	uint8_t FOUND=0;
	if( (  (PB_1 <= 0) && (LAYER->SIGN==0)) || ((PB_1 >= 0) && (LAYER->SIGN!=0))  ){
		_row  = 0;_col  = 0;
		_n    = 0;_m    = 0;
		_kc   = 0;_kr   = 0;
		loop_cnt = 1;
		FOUND=1;
	}else{
		loop_cnt = (LAYER->SIGN==0) ? PB_1 : PB_1 * -1;

		if(loop_cnt<0)loop_cnt=1;

		for(rrr=0; rrr<ROWS ; rrr+=Tr){for(ccc=0; ccc<COLS ; ccc+=Tc){
		for(mmm=0; mmm <MCH  ; mmm +=Tm){for(nnn=0;  nnn <NCH  ; nnn +=Tn){
		for(kkr=0; kkr <KR ; kkr++ ){for(kkc=0; kkc < KC ; kkc++ ){
			if(loop_cnt == temp){
				if(!FOUND){
					FOUND = 1;
					_row  = rrr;_col  = ccc;
					_n    = nnn;_m    = mmm;
					_kc   = kkc;_kr   = kkr;
					_tr = (_row+Tr > ROWS) ? ROWS - _row : Tr;
					_tc = (_col+Tc > COLS) ? COLS - _col : Tc;
					_tn = (_n+Tn > NCH) ? NCH - _n : Tn;
					_tm = (_m+Tm > MCH) ? MCH - _m : Tm;

					ADDR.L = PB+ Aoffset3D(_tr-1,_tc-1,_tm*2 -1 , Tc, 2 * Tm)*SIZE_of_Q15;
					SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
					//ACCUMULATE
					if(PB_L == PB_1){
						OP_L = 1;
						for(sub_op_r = rrr ; sub_op_r < rrr + _tr -1 ; sub_op_r++ ){
							ADDR.L = D_OUT_Ptr + Aoffset3D(sub_op_r,0,1 , COLS, MCH * 2)*SIZE_of_Q15;
							SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
							if(LAYER->SIGN==0){if(PB_L >= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
							else{if(PB_L <= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
						}
						for(sub_op_c = ccc ; sub_op_c < ccc + _tc -1; sub_op_c++ ){
							ADDR.L = D_OUT_Ptr + Aoffset3D(sub_op_r,sub_op_c,1 , COLS, MCH * 2)*SIZE_of_Q15;
							SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
							if(LAYER->SIGN==0){if(PB_L >= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
							else{if(PB_L <= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
						}
						for(sub_op_m = mmm ; sub_op_m < mmm + _tm ; sub_op_m++ ){
							ADDR.L = D_OUT_Ptr + Aoffset3D(sub_op_r,sub_op_c,sub_op_m*2+1 , COLS, MCH * 2)*SIZE_of_Q15;
							SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
							if(LAYER->SIGN==0){if(PB_L >= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
							else{if(PB_L <= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
						}
					}else{
						OP_L = 0;
						//Search PB [_tr][_tc][_tm]
						for(sub_op_r = 0 ; sub_op_r < _tr -1 ; sub_op_r++ ){
							ADDR.L = PB + Aoffset3D(sub_op_r,0,1 , _tc, _tm * 2)*SIZE_of_Q15;
							SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
							if(LAYER->SIGN==0){if(PB_L >= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
							else{if(PB_L <= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
						}if(sub_op_r == _tr)sub_op_r=_tr-1;
						for(sub_op_c = 0 ; sub_op_c < _tc -1 ; sub_op_c++ ){
							ADDR.L = PB + Aoffset3D(sub_op_r,sub_op_c,1 ,  _tc, _tm * 2)*SIZE_of_Q15;
							SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
							if(LAYER->SIGN==0){if(PB_L >= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
							else{if(PB_L <= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
						}if(sub_op_c == _tc)sub_op_c=_tc-1;
						for(sub_op_m = 0 ; sub_op_m < _tm -1 ; sub_op_m++ ){
							ADDR.L = PB + Aoffset3D(sub_op_r,sub_op_c,sub_op_m*2+1 ,  _tc, _tm * 2)*SIZE_of_Q15;
							SPI_READ(&ADDR,(uint8_t*)&PB_L,SIZE_of_Q15);
							if(LAYER->SIGN==0){if(PB_L >= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
							else{if(PB_L <= LL_MARKER){LL_MARKER = PB_L;}else{break;}}
						}
					}
					break;
				}

			}
			else{ temp++;}
		}}}}}}
	}
	if(!FOUND){
		_row  = ROWS;_col  = COLS;
		_n    = NCH;_m    = MCH;
		_kc   = KC;_kr   = KR;
		sub_op_r = _tr;
		sub_op_c = _tc;
		sub_op_m = _tm;
	}


	for(; _row<ROWS ; _row+=Tr){for(; _col<COLS ; _col+=Tc){
	for(; _m <MCH  ; _m +=Tm){for(;  _n <NCH  ; _n +=Tn){
		_tr = (_row+Tr > ROWS) ? ROWS - _row : Tr;
		_tc = (_col+Tc > COLS) ? COLS - _col : Tc;
		_tn = (_n+Tn > NCH) ? NCH - _n : Tn;
		_tm = (_m+Tm > MCH) ? MCH - _m : Tm;
		//Fetch tile to SRAM

		tile_i =(JAP_TILE){.PTR=D_IN_Ptr,
				          .ROWS = ROWS, .COLS= COLS, .NCH = NCH,
						  .KR   = KR,   .KC  = KC,
						  ._row = _row, ._col= _col, ._n=_n,
						  ._row_step = _tr, ._col_step = _tc, ._n_step=_tn};
		//	SRAM_BUFFER
		for(; _kr < KR ; _kr++ ){for( ; _kc < KC ; _kc++ ){
			tile_i._kc = _kc;
			tile_i._kr = _kr;

			if(OP_L==0){

				//Restore
				if( (sub_op_r != 0) || (sub_op_c !=0) ){
					uint32_t DDD;
					DDD = Aoffset3D(sub_op_r,sub_op_c,0 ,  _tc, _tm * 2)*SIZE_of_Q15;
					ADDR.L = PB;
					temp_2 = LEA_MEMORY+lea_o_offset;
					SPI_READ(&ADDR,(uint8_t*)temp_2,DDD);
				}

				JAP_FETCH_INPUT(&tile_i,LEA_MEMORY,D_IN->PG ,LAYER->SIGN ,sub_op_r,sub_op_c); //[R+K-1][C+K-1][tn]

				//Fetch Weight
				ADDR.L =WEIGHT_Ptr + Aoffset4D(_kr,_kc,_n,_m,KC,NCH,MCH)*SIZE_of_Q15;
				srcB =LEA_MEMORY + lea_w_offset;
				//extra kernel;
				length = _tn;
				while(length--){
					// Native_DMA0(srcA,srcB, _tm);
					SPI_READ(&ADDR,(uint8_t*)srcB,_tm*SIZE_of_Q15);
					for(uint16_t i=_tm;i>0;i--){
						srcB[((i<<1)-1)] = 0;
						srcB[((i<<1)-2)] = srcB[i-1];
					}
					ADDR.L  += MCH*SIZE_of_Q15;
					srcB += _tm*2;
				}
				//extra channel
				for(uint16_t i=_tm;i>0;i--){
					srcB[((i<<1)-1)] = dummy_w * loop_cnt ;
					srcB[((i<<1)-2)] = 0;
				}

				if(_tn & 0x1){;}
				else{
					srcB += 2*_tm;
					for(uint16_t i=_tm<<1;i>0;i--)srcB[i-1]=0;
				}

			    mpyParams.srcARows = _tc * _tr - (sub_op_r * _tc + sub_op_c);
			    if(sub_op_r == _tr && sub_op_c == _tc) mpyParams.srcARows=0;
			    mpyParams.srcBCols = _tm*2;
			    if(_tn & 0x1){
				    mpyParams.srcACols = _tn+1;
					mpyParams.srcBRows = _tn+1;
			    }
			    else{
				    mpyParams.srcACols = _tn+2;
					mpyParams.srcBRows = _tn+2;
			    }
			    offset_prog = Aoffset3D(sub_op_r,sub_op_c,0,_tc,_tm*2);
			    JAP_matrix_mpy_q15(&mpyParams,LEA_MEMORY,LEA_MEMORY+lea_w_offset,LEA_MEMORY+lea_o_offset+offset_prog,PB+offset_prog*SIZE_of_Q15);

			    OP_L = 1;
				sub_op_r=_row,sub_op_c=_col,sub_op_m=_m;
			}
			if(OP_L==1){
			    if( (_kr==0) &&  (_kc==0) && (_n==0)){
			    	offset_prog = Aoffset3D(sub_op_r-_row,sub_op_c-_col,(sub_op_m-_m)*2,_tc,_tm*2);
			    	if( (sub_op_r-_row) || (sub_op_c-_col) || (sub_op_m-_m)){
			    		srcA = LEA_MEMORY + lea_o_offset + offset_prog;
			    		ADDR.L = PB + offset_prog*SIZE_of_Q15;
			    		SPI_READ(&ADDR,(uint8_t*)srcA,((2 * _tm * _tr * _tc) - offset_prog)*SIZE_of_Q15);
			    	}

			    	srcA = LEA_MEMORY + lea_o_offset + offset_prog;
			    	ADDR.L = D_OUT_Ptr + Aoffset3D(sub_op_r,sub_op_c,sub_op_m*2,COLS,MCH*2)*SIZE_of_Q15;//*size extra ch

					length_r = _tr-(sub_op_r-_row);
					length_c = _tc-(sub_op_c-_col);
					while(length_r--){
						while(length_c--){
							// Native_DMA0(srcA,srcB,_tm);
							SPI_WRITE(&ADDR,(uint8_t*)srcA,(_tm-(sub_op_m-_m))*2*SIZE_of_Q15);
				    		srcA += (_tm-(sub_op_m-_m))*2;
				    		ADDR.L += (MCH-(sub_op_m-_m))*2*SIZE_of_Q15;
				    		sub_op_m=_m;
						}
						ADDR.L += (COLS - _tc)* MCH*2*SIZE_of_Q15;
						length_c = _tc;
					}
					sub_op_r=_row,sub_op_c=_col,sub_op_m=_m;

			    }else{
			    	offset_prog = Aoffset3D(sub_op_r-_row,sub_op_c-_col,(sub_op_m-_m)*2,_tc,_tm*2);
	    	
			    	if( (sub_op_r-_row) || (sub_op_c-_col) || (sub_op_m-_m)){

			    		ADDR.L = PB + offset_prog *SIZE_of_Q15;
			    		temp_2 = LEA_MEMORY+lea_o_offset + offset_prog;
			    		SPI_READ(&ADDR,(uint8_t*)temp_2, (_tr*_tc*2*_tm - offset_prog) * SIZE_of_Q15);
			    	}
			    	srcA = LEA_MEMORY + lea_o_offset + offset_prog;
			    	ADDR.L = D_OUT_Ptr + Aoffset3D(sub_op_r,sub_op_c,sub_op_m*2,COLS,MCH*2)*SIZE_of_Q15;

			    	length_r = _tr-(sub_op_r-_row);
			    	length_c = _tc-(sub_op_c-_col);
					while(length_r--){
						while(length_c--){

							SPI_READ(&ADDR,(uint8_t*)LEA_MEMORY,(_tm- (sub_op_m-_m))*2*SIZE_of_Q15);
							addParams.length = (_tm- (sub_op_m-_m))*2;
							JAP_ADD(&addParams, srcA, LEA_MEMORY, LEA_MEMORY,ADDR.L);
				    		srcA += (_tm- (sub_op_m-_m))*2;
				    		ADDR.L += (MCH-sub_op_m- _m)*2*SIZE_of_Q15;
				    		sub_op_m = _m;
						}
						ADDR.L += (COLS - _tc)* MCH * 2*SIZE_of_Q15;
						length_c = _tc;
					}

			    }
				OP_L = 0;
				sub_op_r=0,sub_op_c=0,sub_op_m=0;
				loop_cnt++;
			}

		}_kc=0;}_kr=0;

	}_tn=0;}_tm=0;}_col=0;}_row=0;

	LAYER->SIGN = LAYER->SIGN ? 0 : 1 ;


}
