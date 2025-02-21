from gi.repository import GObject
from gi.repository import GLib
from gi.repository import Gtk
from gi.repository import Gst
from ctypes import *

class PlaybackInterface:

    def __init__(self):
        self.playing = False
        self.ispaused = False

        # A free example sound track
        self.uri = "file:///my/crazy/python/2012.mp3"

        # GTK window and widgets
        self.window = Gtk.Window()
        self.window.set_size_request(300,50)
        self.window.set_title("My Gstreamer Player V1.0")

        vbox = Gtk.Box(Gtk.Orientation.HORIZONTAL, 0)
        vbox.set_margin_top(3)
        vbox.set_margin_bottom(3)
        self.window.add(vbox)

        self.playButtonImage = Gtk.Image()
        self.playButtonImage.set_from_icon_name("media-playback-start", Gtk.IconSize.BUTTON)
        self.playButton = Gtk.Button()
        self.playButton.add(self.playButtonImage)
        self.playButton.connect("clicked", self.playToggled)
        Gtk.Box.pack_start(vbox, self.playButton, False, False, 0)

        self.playButtonImage2 = Gtk.Image()
        self.playButtonImage2.set_from_icon_name("media-playback-pause", Gtk.IconSize.BUTTON)
        self.playButton2 = Gtk.Button()
        self.playButton2.add(self.playButtonImage2)
        self.playButton2.connect("clicked", self.pauseToggled)
        self.playButton2.set_sensitive(False)
        Gtk.Box.pack_start(vbox, self.playButton2, False, False, 0)

        self.playButtonImage3 = Gtk.Image()
        self.playButtonImage3.set_from_icon_name("media-seek-forward", Gtk.IconSize.BUTTON)
        self.playButton3 = Gtk.Button()
        self.playButton3.add(self.playButtonImage3)
        self.playButton3.connect("clicked", self.pauseToggled)
        self.playButton3.set_sensitive(False)
        Gtk.Box.pack_start(vbox, self.playButton3, False, False, 0)

        self.slider = Gtk.HScale()
        self.slider.set_margin_left(6)
        self.slider.set_margin_right(6)
        self.slider.set_draw_value(False)
        self.slider.set_range(0, 100)
        self.slider.set_increments(1, 10)
        #self.slider.connect("value-changed", self.sliderValueChanged)
        self.slider.connect("change-value", self.sliderChangeValue)
        Gtk.Box.pack_start(vbox, self.slider, True, True, 0)

        self.label = Gtk.Label(label='0:00')
        self.label.set_margin_left(6)
        self.label.set_margin_right(6)
        Gtk.Box.pack_start(vbox, self.label, False, False, 0)

        self.window.connect("destroy", self.playerQuit)
        self.window.show_all()

        # GStreamer Setup
        Gst.init_check(None)
        self.IS_GST010 = Gst.version()[0] == 0
        self.player = Gst.ElementFactory.make("playbin", "player")
        fakesink = Gst.ElementFactory.make("fakesink", "fakesink")
        self.player.set_property("video-sink", fakesink)
        bus = self.player.get_bus()
        #bus.add_signal_watch_full()
        bus.connect("message", self.on_message)
        self.player.connect("about-to-finish",  self.on_finished)

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.Message.EOS:
                self.player.set_state(Gst.State.NULL)
                self.playing = False
        elif t == Gst.Message.ERROR:
                self.player.set_state(Gst.State.NULL)
                err, debug = message.parse_error()
                print "Error: %s" % err, debug
                self.playing = False

        self.updateButtons()

    def on_finished(self, player):
        self.playing = False
        self.slider.set_value(0)
        self.label.set_text("0:00")
        self.updateButtons()

    def play(self):
        self.player.set_property("uri", self.uri)
        self.player.set_state(Gst.State.PLAYING)
        GObject.timeout_add(1000, self.updateSlider)

    def stop(self):
        self.player.set_state(Gst.State.NULL)
        
    def playToggled(self, w):
        self.slider.set_value(0)
        self.label.set_text("0:00")

        if(self.playing == False):
                self.play()
        else:
                self.stop()

        self.playing=not(self.playing)
        self.updateButtons()

    def pauseToggled(self, w):
        if(self.ispaused == False):
           self.player.set_state(Gst.State.PAUSED)
        else:
           self.player.set_state(Gst.State.PLAYING)
        self.ispaused = not(self.ispaused)

    def sliderValueChanged(self, w):
        print w.get_value()

    def sliderChangeValue(self, w, scroll, value):
        print value
        print Gst.CLOCK_TIME_NONE
        self.player.seek(1.0, Gst.Format.TIME, Gst.SeekFlags.FLUSH, Gst.SeekType.SET,\
                         value * Gst.SECOND, Gst.SeekType.NONE, c_long(Gst.CLOCK_TIME_NONE).value)

    def updateSlider(self):
        if(self.playing == False):
           return False # cancel timeout

        try:
           if self.IS_GST010:
              nanosecs = self.player.query_position(Gst.Format.TIME)[2]
              duration_nanosecs = self.player.query_duration(Gst.Format.TIME)[2]
           else:
              nanosecs = self.player.query_position(Gst.Format.TIME)[1]
              duration_nanosecs = self.player.query_duration(Gst.Format.TIME)[1]

           # block seek handler so we don't seek when we set_value()
           # self.slider.handler_block_by_func(self.on_slider_change)

           duration = float(duration_nanosecs) / Gst.SECOND
           position = float(nanosecs) / Gst.SECOND
           self.slider.set_range(0, duration)
           self.slider.set_value(position)
           self.label.set_text ("%d" % (position / 60) + ":%02d" % (position % 60))

           #self.slider.handler_unblock_by_func(self.on_slider_change)
        
        except Exception as e:
                # pipeline must not be ready and does not know position
                print e
                pass

        return True

    def updateButtons(self):
        if(self.playing == False):
           self.playButtonImage.set_from_icon_name("media-playback-start", Gtk.IconSize.BUTTON)
           self.playButton2.set_sensitive(False)
        else:
           self.playButtonImage.set_from_icon_name("media-playback-stop", Gtk.IconSize.BUTTON)
           self.playButton2.set_sensitive(True)

    def playerQuit(self, w):
        myplayer.player.set_state(Gst.State.NULL)
        Gtk.main_quit()

if __name__ == "__main__":
        myplayer = PlaybackInterface()
        Gtk.main()
