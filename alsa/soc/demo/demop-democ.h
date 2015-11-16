#ifndef _DEMOP_DEMOC_H_
#define _DEMOP_DEMOC_H_ 1

#include "democ.h"

struct demop_democ_platform_data {
	int l3_clk;
	int l3_mode;
	int l3_data;
	void (*power) (int);
	int model;
};

#endif
