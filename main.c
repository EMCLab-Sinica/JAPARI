



#include <msp430.h>


#include "japari.h"
#include "myuart.h"
volatile uint32_t cycleCount[3];
volatile uint32_t cycleCount2;


#define ConV_2  0x7CB40
#define ConV_2_b  0x7CAC0
#define  BUFF_L1  0x3FEAC
#define  BUFF_L3  0x2E52C
#define  PBUFF_L3  0x29804

JAP_LAYER SIG[1]={
		(JAP_LAYER){
			.fun       = JAP_CONV,
			.DATA_IN   = (JAP_DATA){.DATA_Ptr=BUFF_L1,.W =15,.H=15,.CH=32,.PG=1,},
			.DATA_OUT  = (JAP_DATA){.DATA_Ptr=BUFF_L3,.W =13,.H=13,.CH=64,.PG=1,},
			.PARA      = (JAP_PARA){.KERNEL_W=3,.KERNEL_H=3,  .CH_IN = 32, .CH_OUT=64, .WEIGHT = ConV_2, .BIAS = ConV_2_b},
			.SIGN      = 0,
			.BATCH     = BA_SIZE,
			.BUFFER_Ptr= PBUFF_L3}
};
#pragma PERSISTENT(network)
JAP_NETWORK network={
	.LAYERS 	  = SIG,
	.TOTAL_LAYERS = 1,
	.FOOTPRINT 	  = 0
};


#pragma PERSISTENT(ERASE_FLAG)
uint16_t ERASE_FLAG = 0;

#pragma PERSISTENT(IF_CNT)
uint16_t IF_CNT = 0;

uint16_t UART_FG=0;
unsigned int FreqLevel = 9;
int uartsetup=0;



int boardSetup(){
    WDTCTL = WDTPW + WDTHOLD;    /* disable watchdog timer  */

    P1DIR=0xff;P1OUT=0x00;
    P2DIR=0xff;P2OUT=0x00;
    P3DIR=0xff;P3OUT=0x00;
    P4DIR=0xff;P4OUT=0x00;
    P5DIR=0xff;P5OUT=0x00;
    P6DIR=0xff;P6OUT=0x00;
    P7DIR=0xff;P7OUT=0x00;
    P8DIR=0xff;P8OUT=0x00;
    PADIR=0xff;PAOUT=0x00;
    PBDIR=0xff;PBOUT=0x00;
    PCDIR=0xff;PCOUT=0x00;
    PDDIR=0xff;PDOUT=0x00;

    P8DIR=0xfd;P8REN=GPIO_PIN1;


    uartsetup=0;
    setFrequency(8);
    // setup uart
    CS_initClockSignal( CS_SMCLK, CS_DCOCLK_SELECT, CS_CLOCK_DIVIDER_1 );

    CSCTL0_H = CSKEY_H;                     // Unlock CS registers
    CSCTL1 = DCOFSEL_0;                     // Set DCO to 1MHz
    // Set SMCLK = MCLK = DCO, ACLK = VLOCLK
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    // Per Device Errata set divider to 4 before changing frequency to
    // prevent out of spec operation from overshoot transient
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;   // Set all corresponding clk sources to divide by 4 for errata
    CSCTL1 = DCOFSEL_4 | DCORSEL;

    __delay_cycles(80);
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;   // Set all dividers to 1 for 16MHz operation
    CSCTL0_H = 0;


    PMM_unlockLPM5();

    __delay_cycles(100);

    _enable_interrupt();

    msp_lea_ifg = 0;
    msp_lea_locked = 0;

    /* Initialize LEA registers. */
    LEACNF0 = LEALPR | LEAILPM;
    LEACNF1 = 0;
    LEACNF2 = LEAMT >> 2;
    LEAPMS1 = 0;
    LEAPMS0 = 0;
    LEAPMDST = 0;
    LEAPMCTL |= LEACMDEN;

    LEACMCTL = 0;

    return 0;
};

void main(void) {

	boardSetup();

	UART_FG=0;
	__delay_cycles(450000);
	initSPI();

if(ERASE_FLAG==0){
    eraseFRAM();
    ERASE_FLAG = 1;
    P1DIR |= 0x01;
    P1OUT |= 0x1;
    while(1);
}


while(P8IN & GPIO_PIN1){
	if(!UART_FG){uartinit();UART_FG=1;}
	_DBGUART("\r\nCMD,cnt,%d , L: %d\r\n",IF_CNT , network.FOOTPRINT);

    for(int i = 0 ;i<5;i++)
	__delay_cycles(5000000);
}

    while(1){
		JAP_INFERENCE(&network);
		IF_CNT++;
		P1OUT ^= 0x01;
    }


}

