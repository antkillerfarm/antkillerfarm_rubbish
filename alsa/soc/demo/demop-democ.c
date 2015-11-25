
#include <linux/clk.h>
#include <linux/gpio.h>
#include <linux/module.h>

#include <sound/soc.h>
#include "demop-democ.h"

#include "demop-i2s.h"

#define pr_debug(format, ...) printk(KERN_INFO format, ## __VA_ARGS__)

/* #define ENFORCE_RATES 1 */

static struct clk *xtal;
static struct clk *pclk;
/* this is need because we don't have a place where to keep the
 * pointers to the clocks in each substream. We get the clocks only
 * when we are actually using them so we don't block stuff like
 * frequency change or oscillator power-off */
static int clk_users;
static DEFINE_MUTEX(clk_lock);

static unsigned int rates[33 * 2];
#ifdef ENFORCE_RATES
static struct snd_pcm_hw_constraint_list hw_constraints_rates = {
	.count	= ARRAY_SIZE(rates),
	.list	= rates,
	.mask	= 0,
};
#endif

static struct platform_device *demop_democ_snd_device;

static int demop_democ_startup(struct snd_pcm_substream *substream)
{
	pr_debug("%s %d\n", __func__, clk_users);

	return 0;
}

static void demop_democ_shutdown(struct snd_pcm_substream *substream)
{
	pr_debug("%s %d\n", __func__, clk_users);
}

static int demop_democ_hw_params(struct snd_pcm_substream *substream,
					struct snd_pcm_hw_params *params)
{
	pr_debug("%s %d\n", __func__, clk_users);

	return 0;
}

static struct snd_soc_ops demop_democ_ops = {
	.startup = demop_democ_startup,
	.shutdown = demop_democ_shutdown,
	.hw_params = demop_democ_hw_params,
};

static struct snd_soc_dai_link demop_democ_dai_link = {
	.name = "DEMOC",
	.stream_name = "DEMOC",
	.codec_name = "democ-codec",
	.codec_dai_name = "democ-hifi",
	.cpu_dai_name = "demop-iis",
	.ops = &demop_democ_ops,
	.platform_name	= "demop-iis",
};

static struct snd_soc_card snd_soc_demop_democ = {
	.name = "DEMOP_DEMOC",
	.owner = THIS_MODULE,
	.dai_link = &demop_democ_dai_link,
	.num_links = 1,
};

static struct demop_democ_platform_data *demop_democ_l3_pins;

static void setdat(int v)
{
  //gpio_set_value(demop_democ_l3_pins->l3_data, v > 0);
}

static void setclk(int v)
{
  //gpio_set_value(demop_democ_l3_pins->l3_clk, v > 0);
}

static void setmode(int v)
{
  //gpio_set_value(demop_democ_l3_pins->l3_mode, v > 0);
}

/* FIXME - This must be codec platform data but in which board file ?? */

static int demop_democ_setup_pin(int pin, char *fun)
{
	pr_debug("%s %d\n", __func__, clk_users);

	return 0;
}

static int demop_democ_probe(struct platform_device *pdev)
{
	int ret;

	printk(KERN_INFO "DEMOP_DEMOC SoC Audio driver\n");


	demop_democ_snd_device = platform_device_alloc("soc-audio", -1);
	if (!demop_democ_snd_device) {
		printk(KERN_ERR "DEMOP_DEMOC SoC Audio: "
		       "Unable to register\n");
		return -ENOMEM;
	}

	platform_set_drvdata(demop_democ_snd_device,
			     &snd_soc_demop_democ);
	ret = platform_device_add(demop_democ_snd_device);
	if (ret) {
		printk(KERN_ERR "DEMOP_DEMOC SoC Audio: Unable to add\n");
		platform_device_put(demop_democ_snd_device);
	}

	return ret;
}

static int demop_democ_remove(struct platform_device *pdev)
{
	pr_debug("%s %d\n", __func__, clk_users);

	platform_device_unregister(demop_democ_snd_device);

	return 0;
}

static struct platform_driver demop_democ_driver = {
	.probe  = demop_democ_probe,
	.remove = demop_democ_remove,
	.driver = {
		.name = "demop-democ",
		.owner = THIS_MODULE,
	},
};

//module_platform_driver(demop_democ_driver);

struct platform_device demop_device_iis = {
	.name		= "demop-iis",
	.id		= -1,
};

struct platform_device demop_codec = {
	.name		= "democ-codec",
	.id		= -1,
};

struct platform_device demop_audio = {
	.name		= "demop-democ",
	.id		= -1,
};

static struct platform_device *demop_devices[] __initdata = {
	&demop_device_iis,
	&demop_codec,
	&demop_audio,
};

static int __init demop_democ_driver_init(void)
{
#if 0
	int ret;
	unsigned short ipsel;

	/* enable both AC97 controllers in pinmux reg */
	ipsel = __raw_readw(IPSEL);
	__raw_writew(ipsel | (3 << 10), IPSEL);

	ret = -ENOMEM;
	sh7760_ac97_snd_device = platform_device_alloc("soc-audio", -1);
	if (!sh7760_ac97_snd_device)
		goto out;

	platform_set_drvdata(sh7760_ac97_snd_device,
			     &sh7760_ac97_soc_machine);
	ret = platform_device_add(sh7760_ac97_snd_device);

	if (ret)
		platform_device_put(sh7760_ac97_snd_device);

out:
	return ret;
#endif
	platform_add_devices(demop_devices, ARRAY_SIZE(demop_devices));
	return platform_driver_register(&demop_democ_driver);
}

static void __exit demop_democ_driver_exit(void)
{
	//platform_device_unregister(sh7760_ac97_snd_device);
        platform_driver_unregister(&demop_democ_driver);
}

module_init(demop_democ_driver_init);
module_exit(demop_democ_driver_exit);

MODULE_AUTHOR("Antkillerfarm");
MODULE_DESCRIPTION("DEMOP_DEMOC ALSA SoC audio driver");
MODULE_LICENSE("GPL");
