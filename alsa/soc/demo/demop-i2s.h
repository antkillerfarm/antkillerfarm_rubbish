/*
 * s3c24xx-i2s.c  --  ALSA Soc Audio Layer
 *
 * Copyright 2005 Wolfson Microelectronics PLC.
 * Author: Graeme Gregory
 *         graeme.gregory@wolfsonmicro.com or linux@wolfsonmicro.com
 *
 *  This program is free software; you can redistribute  it and/or modify it
 *  under  the terms of  the GNU General  Public License as published by the
 *  Free Software Foundation;  either version 2 of the  License, or (at your
 *  option) any later version.
 *
 *  Revision history
 *    10th Nov 2006   Initial version.
 */

#ifndef DEMOPI2S_H_
#define DEMOPI2S_H_

/* clock sources */
#define DEMOP_CLKSRC_PCLK 0
#define DEMOP_CLKSRC_MPLL 1

/* Clock dividers */
#define DEMOP_DIV_MCLK	0
#define DEMOP_DIV_BCLK	1
#define DEMOP_DIV_PRESCALER	2

/* prescaler */
#define DEMOP_PRESCALE(a,b) \
	(((a - 1) << S3C2410_IISPSR_INTSHIFT) | ((b - 1) << S3C2410_IISPSR_EXTSHFIT))

u32 demop_i2s_get_clockrate(void);

#endif /*DEMOPI2S_H_*/
