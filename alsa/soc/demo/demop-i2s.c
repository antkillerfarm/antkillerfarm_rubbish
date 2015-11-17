/*
 * s3c24xx-i2s.c  --  ALSA Soc Audio Layer
 *
 * (c) 2006 Wolfson Microelectronics PLC.
 * Graeme Gregory graeme.gregory@wolfsonmicro.com or linux@wolfsonmicro.com
 *
 * Copyright 2004-2005 Simtec Electronics
 *	http://armlinux.simtec.co.uk/
 *	Ben Dooks <ben@simtec.co.uk>
 *
 *  This program is free software; you can redistribute  it and/or modify it
 *  under  the terms of  the GNU General  Public License as published by the
 *  Free Software Foundation;  either version 2 of the  License, or (at your
 *  option) any later version.
 */

#include <linux/delay.h>
#include <linux/clk.h>
#include <linux/io.h>
#include <linux/gpio.h>
#include <linux/module.h>

#include <sound/soc.h>
#include <sound/pcm_params.h>

#include "demop-i2s.h"
#if 0
static struct s3c_dma_params demop_i2s_pcm_stereo_out = {
	.channel	= DMACH_I2S_OUT,
	.ch_name	= "tx",
	.dma_size	= 2,
};

static struct s3c_dma_params demop_i2s_pcm_stereo_in = {
	.channel	= DMACH_I2S_IN,
	.ch_name	= "rx",
	.dma_size	= 2,
};
#endif
struct demop_i2s_info {
	void __iomem	*regs;
	struct clk	*iis_clk;
	u32		iiscon;
	u32		iismod;
	u32		iisfcon;
	u32		iispsr;
};
static struct demop_i2s_info demop_i2s;

static void demop_snd_txctrl(int on)
{
	pr_debug("Entered %s\n", __func__);
}

static void demop_snd_rxctrl(int on)
{
	pr_debug("Entered %s\n", __func__);
}

/*
 * Wait for the LR signal to allow synchronisation to the L/R clock
 * from the codec. May only be needed for slave mode.
 */
static int demop_snd_lrsync(void)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

/*
 * Check whether CPU is the master or slave
 */
static inline int demop_snd_is_clkmaster(void)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

/*
 * Set Demop I2S DAI format
 */
static int demop_i2s_set_fmt(struct snd_soc_dai *cpu_dai,
		unsigned int fmt)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

static int demop_i2s_hw_params(struct snd_pcm_substream *substream,
				 struct snd_pcm_hw_params *params,
				 struct snd_soc_dai *dai)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

static int demop_i2s_trigger(struct snd_pcm_substream *substream, int cmd,
			       struct snd_soc_dai *dai)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

/*
 * Set Demop Clock source
 */
static int demop_i2s_set_sysclk(struct snd_soc_dai *cpu_dai,
	int clk_id, unsigned int freq, int dir)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

/*
 * Set Demop Clock dividers
 */
static int demop_i2s_set_clkdiv(struct snd_soc_dai *cpu_dai,
	int div_id, int div)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

/*
 * To avoid duplicating clock code, allow machine driver to
 * get the clockrate from here.
 */
u32 demop_i2s_get_clockrate(void)
{
	return 100;
}
EXPORT_SYMBOL_GPL(demop_i2s_get_clockrate);

static int demop_i2s_probe(struct snd_soc_dai *dai)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

#ifdef CONFIG_PM
static int demop_i2s_suspend(struct snd_soc_dai *cpu_dai)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}

static int demop_i2s_resume(struct snd_soc_dai *cpu_dai)
{
	pr_debug("Entered %s\n", __func__);

	return 0;
}
#else
#define demop_i2s_suspend NULL
#define demop_i2s_resume NULL
#endif


#define DEMOP_I2S_RATES \
	(SNDRV_PCM_RATE_8000 | SNDRV_PCM_RATE_11025 | SNDRV_PCM_RATE_16000 | \
	SNDRV_PCM_RATE_22050 | SNDRV_PCM_RATE_32000 | SNDRV_PCM_RATE_44100 | \
	SNDRV_PCM_RATE_48000 | SNDRV_PCM_RATE_88200 | SNDRV_PCM_RATE_96000)

static const struct snd_soc_dai_ops demop_i2s_dai_ops = {
	.trigger	= demop_i2s_trigger,
	.hw_params	= demop_i2s_hw_params,
	.set_fmt	= demop_i2s_set_fmt,
	.set_clkdiv	= demop_i2s_set_clkdiv,
	.set_sysclk	= demop_i2s_set_sysclk,
};

static struct snd_soc_dai_driver demop_i2s_dai = {
	.probe = demop_i2s_probe,
	.suspend = demop_i2s_suspend,
	.resume = demop_i2s_resume,
	.playback = {
		.channels_min = 2,
		.channels_max = 2,
		.rates = DEMOP_I2S_RATES,
		.formats = SNDRV_PCM_FMTBIT_S8 | SNDRV_PCM_FMTBIT_S16_LE,},
	.capture = {
		.channels_min = 2,
		.channels_max = 2,
		.rates = DEMOP_I2S_RATES,
		.formats = SNDRV_PCM_FMTBIT_S8 | SNDRV_PCM_FMTBIT_S16_LE,},
	.ops = &demop_i2s_dai_ops,
};

static const struct snd_soc_component_driver demop_i2s_component = {
	.name		= "demop-i2s",
};

static int demop_iis_dev_probe(struct platform_device *pdev)
{
	int ret = 0;
	struct resource *res;

	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res) {
		dev_err(&pdev->dev, "Can't get IO resource.\n");
		return -ENOENT;
	}

	ret = devm_snd_soc_register_component(&pdev->dev,
			&demop_i2s_component, &demop_i2s_dai, 1);
	if (ret) {
		pr_err("failed to register the dai\n");
		return ret;
	}

	return ret;
}

static struct platform_driver demop_iis_driver = {
	.probe  = demop_iis_dev_probe,
	.driver = {
		.name = "demop-iis",
		.owner = THIS_MODULE,
	},
};

module_platform_driver(demop_iis_driver);

/* Module information */
MODULE_AUTHOR("Ben Dooks, <ben@simtec.co.uk>");
MODULE_DESCRIPTION("demop I2S SoC Interface");
MODULE_LICENSE("GPL");
MODULE_ALIAS("platform:demop-iis");
