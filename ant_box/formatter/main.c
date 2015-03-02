#include <gtk/gtk.h>
#include <string.h>

typedef struct{
	GtkEntry *entry_con;
	GtkEntry *entry_crc;
	GtkButton *btn_choose_file;
}MainWindowSubWidget;

MainWindowSubWidget main_window_sub_widget;

G_MODULE_EXPORT void do_choose_file_button_clicked(GtkButton *button, gpointer data)
{

}

G_MODULE_EXPORT void do_calc_button_clicked(GtkButton *button, gpointer data)
{

}

int main(int argc, char *argv[])
{
	GtkWidget *window = NULL;
	GtkBuilder *builder;
	GError *err = NULL;
	GtkEntry *entry;
	GtkButton *button;

	gtk_init(NULL, NULL);

	builder = gtk_builder_new();
	gtk_builder_add_from_file(builder, "formatter.glade", &err);

	gtk_builder_connect_signals(builder, NULL);
	window = GTK_WIDGET(gtk_builder_get_object(builder, "main_window"));

	//entry = GTK_ENTRY(gtk_builder_get_object(builder, "entry_src"));
	//main_window_sub_widget.entry_con = entry;
	//entry = GTK_ENTRY(gtk_builder_get_object(builder, "entry_dest"));
	//main_window_sub_widget.entry_crc = entry;
	//button = GTK_BUTTON(gtk_builder_get_object(builder, "btn_choose_file"));
	//main_window_sub_widget.btn_choose_file = button;

	g_signal_connect(window, "destroy",
			 G_CALLBACK (gtk_main_quit), &window);

	g_object_unref(G_OBJECT(builder));

	gtk_widget_show_all(window);
	gtk_main();
	return 0;
}
