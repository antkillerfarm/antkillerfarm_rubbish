/*
 * uda134x.c  --  UDA134X ALSA SoC Codec driver
 *
 * Modifications by Christian Pellegrin <chripell@evolware.org>
 *
 * Copyright 2007 Dension Audio Systems Ltd.
 * Author: Zoltan Devai
 *
 * Based on the WM87xx drivers by Liam Girdwood and Richard Purdie
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#include <linux/module.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <sound/pcm.h>
#include <sound/pcm_params.h>
#include <sound/soc.h>
#include <sound/initval.h>

#include <sound/l3.h>

#include "democ.h"


#define DEMOC_RATES SNDRV_PCM_RATE_8000_48000
#define DEMOC_FORMATS (SNDRV_PCM_FMTBIT_S8 | SNDRV_PCM_FMTBIT_S16_LE | \
		SNDRV_PCM_FMTBIT_S18_3LE | SNDRV_PCM_FMTBIT_S20_3LE)

struct democ_priv {
	int sysclk;
	int dai_fmt;

	struct snd_pcm_substream *master_substream;
	struct snd_pcm_substream *slave_substream;
};

/* In-data addresses are hard-coded into the reg-cache values */
static const char democ_reg[DEMOC_REGS_NUM] = {
	/* Extended address registers */
	0x04, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
	/* Status, data regs */
	0x00, 0x83, 0x00, 0x40, 0x80, 0xC0, 0x00,
};

/*
 * The codec has no support for reading its registers except for peak level...
 */
static inline unsigned int democ_read_reg_cache(struct snd_soc_codec *codec,
	unsigned int reg)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

/*
 * Write the register cache
 */
static inline void democ_write_reg_cache(struct snd_soc_codec *codec,
	u8 reg, unsigned int value)
{
        pr_debug("Entered %s\n", __func__);
}

/*
 * Write to the democ registers
 *
 */
static int democ_write(struct snd_soc_codec *codec, unsigned int reg,
	unsigned int value)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static inline void democ_reset(struct snd_soc_codec *codec)
{
        pr_debug("Entered %s\n", __func__);
}

static int democ_mute(struct snd_soc_dai *dai, int mute)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static int democ_startup(struct snd_pcm_substream *substream,
	struct snd_soc_dai *dai)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static void democ_shutdown(struct snd_pcm_substream *substream,
	struct snd_soc_dai *dai)
{
        pr_debug("Entered %s\n", __func__);
}

static int democ_hw_params(struct snd_pcm_substream *substream,
	struct snd_pcm_hw_params *params,
	struct snd_soc_dai *dai)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static int democ_set_dai_sysclk(struct snd_soc_dai *codec_dai,
				  int clk_id, unsigned int freq, int dir)
{
        pr_debug("Entered %s\n", __func__);

	return -EINVAL;
}

static int democ_set_dai_fmt(struct snd_soc_dai *codec_dai,
			       unsigned int fmt)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static int democ_set_bias_level(struct snd_soc_codec *codec,
				  enum snd_soc_bias_level level)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static const char *democ_dsp_setting[] = {"Flat", "Minimum1",
					    "Minimum2", "Maximum"};
static const char *democ_deemph[] = {"None", "32Khz", "44.1Khz", "48Khz"};
static const char *democ_mixmode[] = {"Differential", "Analog1",
					"Analog2", "Both"};

static const struct soc_enum democ_mixer_enum[] = {
SOC_ENUM_SINGLE(DEMOC_DATA010, 0, 0x04, democ_dsp_setting),
SOC_ENUM_SINGLE(DEMOC_DATA010, 3, 0x04, democ_deemph),
SOC_ENUM_SINGLE(DEMOC_EA010, 0, 0x04, democ_mixmode),
};

static const struct snd_kcontrol_new democ_snd_controls[] = {
SOC_SINGLE("Master Playback Volume", DEMOC_DATA000, 0, 0x3F, 1),
SOC_SINGLE("Capture Volume", DEMOC_EA010, 2, 0x07, 0),
SOC_SINGLE("Analog1 Volume", DEMOC_EA000, 0, 0x1F, 1),
SOC_SINGLE("Analog2 Volume", DEMOC_EA001, 0, 0x1F, 1),

SOC_SINGLE("Mic Sensitivity", DEMOC_EA010, 2, 7, 0),
SOC_SINGLE("Mic Volume", DEMOC_EA101, 0, 0x1F, 0),

SOC_SINGLE("Tone Control - Bass", DEMOC_DATA001, 2, 0xF, 0),
SOC_SINGLE("Tone Control - Treble", DEMOC_DATA001, 0, 3, 0),

SOC_ENUM("Sound Processing Filter", democ_mixer_enum[0]),
SOC_ENUM("PCM Playback De-emphasis", democ_mixer_enum[1]),
SOC_ENUM("Input Mux", democ_mixer_enum[2]),

SOC_SINGLE("AGC Switch", DEMOC_EA100, 4, 1, 0),
SOC_SINGLE("AGC Target Volume", DEMOC_EA110, 0, 0x03, 1),
SOC_SINGLE("AGC Timing", DEMOC_EA110, 2, 0x07, 0),

SOC_SINGLE("DAC +6dB Switch", DEMOC_STATUS1, 6, 1, 0),
SOC_SINGLE("ADC +6dB Switch", DEMOC_STATUS1, 5, 1, 0),
SOC_SINGLE("ADC Polarity Switch", DEMOC_STATUS1, 4, 1, 0),
SOC_SINGLE("DAC Polarity Switch", DEMOC_STATUS1, 3, 1, 0),
SOC_SINGLE("Double Speed Playback Switch", DEMOC_STATUS1, 2, 1, 0),
SOC_SINGLE("DC Filter Enable Switch", DEMOC_STATUS0, 0, 1, 0),
};

#if 0
/* UDA1341 has the DAC/ADC power down in STATUS1 */
static const struct snd_soc_dapm_widget democ_dapm_widgets[] = {
	SND_SOC_DAPM_DAC("DAC", "Playback", DEMOC_STATUS1, 0, 0),
	SND_SOC_DAPM_ADC("ADC", "Capture", DEMOC_STATUS1, 1, 0),
};
#endif

/* Common DAPM widgets */
static const struct snd_soc_dapm_widget democ_dapm_widgets[] = {
	SND_SOC_DAPM_INPUT("VINL1"),
	SND_SOC_DAPM_INPUT("VINR1"),
	SND_SOC_DAPM_INPUT("VINL2"),
	SND_SOC_DAPM_INPUT("VINR2"),
	SND_SOC_DAPM_OUTPUT("VOUTL"),
	SND_SOC_DAPM_OUTPUT("VOUTR"),
};

static const struct snd_soc_dapm_route democ_dapm_routes[] = {
	{ "ADC", NULL, "VINL1" },
	{ "ADC", NULL, "VINR1" },
	{ "ADC", NULL, "VINL2" },
	{ "ADC", NULL, "VINR2" },
	{ "VOUTL", NULL, "DAC" },
	{ "VOUTR", NULL, "DAC" },
};

static const struct snd_soc_dai_ops democ_dai_ops = {
	.startup	= democ_startup,
	.shutdown	= democ_shutdown,
	.hw_params	= democ_hw_params,
	.digital_mute	= democ_mute,
	.set_sysclk	= democ_set_dai_sysclk,
	.set_fmt	= democ_set_dai_fmt,
};

static struct snd_soc_dai_driver democ_dai = {
	.name = "democ-hifi",
	/* playback capabilities */
	.playback = {
		.stream_name = "Playback",
		.channels_min = 1,
		.channels_max = 2,
		.rates = DEMOC_RATES,
		.formats = DEMOC_FORMATS,
	},
	/* capture capabilities */
	.capture = {
		.stream_name = "Capture",
		.channels_min = 1,
		.channels_max = 2,
		.rates = DEMOC_RATES,
		.formats = DEMOC_FORMATS,
	},
	/* pcm operations */
	.ops = &democ_dai_ops,
};

static int democ_soc_probe(struct snd_soc_codec *codec)
{

	printk(KERN_INFO "DEMOC SoC Audio Codec\n");

	return 0;
}

/* power down chip */
static int democ_soc_remove(struct snd_soc_codec *codec)
{
  //struct democ_priv *democ = snd_soc_codec_get_drvdata(codec);

  //kfree(democ);
	return 0;
}

#if defined(CONFIG_PM)
static int democ_soc_suspend(struct snd_soc_codec *codec)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}

static int democ_soc_resume(struct snd_soc_codec *codec)
{
        pr_debug("Entered %s\n", __func__);

	return 0;
}
#else
#define democ_soc_suspend NULL
#define democ_soc_resume NULL
#endif /* CONFIG_PM */

static struct snd_soc_codec_driver soc_codec_dev_democ = {
	.probe =        democ_soc_probe,
	.remove =       democ_soc_remove,
	.suspend =      democ_soc_suspend,
	.resume =       democ_soc_resume,
	.reg_cache_size = sizeof(democ_reg),
	.reg_word_size = sizeof(u8),
	.reg_cache_default = democ_reg,
	.reg_cache_step = 1,
	.read = democ_read_reg_cache,
	.write = democ_write,
	.set_bias_level = democ_set_bias_level,
	.dapm_widgets = democ_dapm_widgets,
	.num_dapm_widgets = ARRAY_SIZE(democ_dapm_widgets),
	.dapm_routes = democ_dapm_routes,
	.num_dapm_routes = ARRAY_SIZE(democ_dapm_routes),
};

static int democ_codec_probe(struct platform_device *pdev)
{
	return snd_soc_register_codec(&pdev->dev,
			&soc_codec_dev_democ, &democ_dai, 1);
}

static int democ_codec_remove(struct platform_device *pdev)
{
	snd_soc_unregister_codec(&pdev->dev);
	return 0;
}

static struct platform_driver democ_codec_driver = {
	.driver = {
		.name = "democ-codec",
		.owner = THIS_MODULE,
	},
	.probe = democ_codec_probe,
	.remove = democ_codec_remove,
};

module_platform_driver(democ_codec_driver);

MODULE_DESCRIPTION("DEMOC ALSA soc codec driver");
MODULE_AUTHOR("Zoltan Devai, Christian Pellegrin <chripell@evolware.org>");
MODULE_LICENSE("GPL");
