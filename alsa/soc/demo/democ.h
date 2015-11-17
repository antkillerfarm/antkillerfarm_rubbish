#ifndef _DEMOC_CODEC_H
#define _DEMOC_CODEC_H

#define DEMOC_L3ADDR	5
#define DEMOC_DATA0_ADDR	((DEMOC_L3ADDR << 2) | 0)
#define DEMOC_DATA1_ADDR	((DEMOC_L3ADDR << 2) | 1)
#define DEMOC_STATUS_ADDR	((DEMOC_L3ADDR << 2) | 2)

#define DEMOC_EXTADDR_PREFIX	0xC0
#define DEMOC_EXTDATA_PREFIX	0xE0

/* DEMOC registers */
#define DEMOC_EA000	0
#define DEMOC_EA001	1
#define DEMOC_EA010	2
#define DEMOC_EA011	3
#define DEMOC_EA100	4
#define DEMOC_EA101	5
#define DEMOC_EA110	6
#define DEMOC_EA111	7
#define DEMOC_STATUS0 8
#define DEMOC_STATUS1 9
#define DEMOC_DATA000 10
#define DEMOC_DATA001 11
#define DEMOC_DATA010 12
#define DEMOC_DATA011 13
#define DEMOC_DATA1   14

#define DEMOC_REGS_NUM 15

#define STATUS0_DAIFMT_MASK (~(7<<1))
#define STATUS0_SYSCLK_MASK (~(3<<4))

#if 0
struct uda134x_platform_data {
	struct l3_pins l3;
	void (*power) (int);
	int model;
	/*
	  ALSA SOC usually puts the device in standby mode when it's not used
	  for sometime. If you unset is_powered_on_standby the driver will
	  turn off the ADC/DAC when this callback is invoked and turn it back
	  on when needed. Unfortunately this will result in a very light bump
	  (it can be audible only with good earphones). If this bothers you
	  set is_powered_on_standby, you will have slightly higher power
	  consumption. Please note that sending the L3 command for ADC is
	  enough to make the bump, so it doesn't make difference if you
	  completely take off power from the codec.
	*/
	int is_powered_on_standby;
#define DEMOC_UDA1340 1
#define DEMOC_UDA1341 2
#define DEMOC_UDA1344 3
#define DEMOC_UDA1345 4
};
#endif

#endif
