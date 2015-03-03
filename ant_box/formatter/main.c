#include <gtk/gtk.h>
#include <string.h>

typedef struct{
	GtkEntry *entry_file_src;
	GtkEntry *entry_file_dest;
}MainWindowSubWidget;

MainWindowSubWidget main_window_sub_widget;

G_MODULE_EXPORT void do_btn_file_src_clicked(GtkButton *button, gpointer data)
{
	GtkWidget *dialog;
	GtkWindow *window = (GtkWindow *)data;

	dialog = gtk_file_chooser_dialog_new ("Open File",
					      window,
					      GTK_FILE_CHOOSER_ACTION_OPEN,
					      GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
					      GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
					      NULL);

	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
	{
		gchar *filename;

		filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
		gtk_entry_set_text(main_window_sub_widget.entry_file_src, filename);
		gtk_entry_set_text(main_window_sub_widget.entry_file_dest, filename);
		g_print("IPhone 5S!%s\r\n",
			filename);
		g_free (filename);
	}

	gtk_widget_destroy (dialog);
}

G_MODULE_EXPORT void do_btn_file_dest_clicked(GtkButton *button, gpointer data)
{
	GtkWidget *dialog;
	GtkWindow *window = (GtkWindow *)data;

	dialog = gtk_file_chooser_dialog_new ("Open File",
					      window,
					      GTK_FILE_CHOOSER_ACTION_OPEN,
					      GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
					      GTK_STOCK_OPEN, GTK_RESPONSE_ACCEPT,
					      NULL);

	if (gtk_dialog_run (GTK_DIALOG (dialog)) == GTK_RESPONSE_ACCEPT)
	{
		gchar *filename;

		filename = gtk_file_chooser_get_filename (GTK_FILE_CHOOSER (dialog));
		gtk_entry_set_text(main_window_sub_widget.entry_file_dest, filename);
		g_free (filename);
	}

	gtk_widget_destroy (dialog);
}

G_MODULE_EXPORT void do_btn_format_clicked(GtkButton *button, gpointer data)
{
	gchar buf[2046];
	gchar *buf_in;
	gint len;

	const gchar * content_src = gtk_entry_get_text(main_window_sub_widget.entry_file_src);
	GFile *file_src = g_file_new_for_path(content_src);
	const gchar * content_dest = gtk_entry_get_text(main_window_sub_widget.entry_file_dest);
	GFile *file_dest = g_file_new_for_path(content_dest);

	GFileInputStream *f_in_stream = g_file_read(file_src, NULL, NULL);
	gint file_len;
	if (f_in_stream != NULL)
	{
			
		g_input_stream_read_all(G_INPUT_STREAM(f_in_stream), buf, 
					2046, &file_len, NULL, NULL);
		buf_in = buf;
		len = file_len;
		//g_print("Jobs!%d/%s\r\n", len, buf_in);
		g_input_stream_close(G_INPUT_STREAM(f_in_stream), NULL, NULL);
		g_object_unref(f_in_stream);
	}
	g_object_unref(file_src);

        gint i, j = 0;
        gint data0, flag = 0;
	//len = strlen(buf_in);
	//g_print("Woz!%d\r\n", len);
	for (i = 0; i < len; i++)
	{
		if (g_ascii_isxdigit(buf_in[i]))
		{
			if (flag == 0)
			{
				data0 = 0;
				data0 = (g_ascii_xdigit_value(buf_in[i])) << 4;
				flag = 1;
			}
			else
			{
				data0 |= g_ascii_xdigit_value(buf_in[i]);
				buf_in[j] = data0;
				j++;
				flag = 0;
			}
		}
	}
	len = j;

	GFileOutputStream *f_out_stream = g_file_replace(file_dest, NULL, FALSE, G_FILE_CREATE_NONE, NULL, NULL);
	if (f_out_stream != NULL)
	{
			
		g_output_stream_write_all(G_OUTPUT_STREAM(f_out_stream), buf, 
					len, &file_len, NULL, NULL);
		//buf_in = buf;
		//len = file_len;
		//g_print("Jobs!%d/%s\r\n", len, buf_in);
		g_output_stream_close(G_OUTPUT_STREAM(f_out_stream), NULL, NULL);
		g_object_unref(f_out_stream);
	}
	g_object_unref(file_dest);

}

G_MODULE_EXPORT void do_btn_exit_clicked(GtkButton *button, gpointer data)
{
	gtk_main_quit();
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

	entry = GTK_ENTRY(gtk_builder_get_object(builder, "entry_file_src"));
	main_window_sub_widget.entry_file_src = entry;
	entry = GTK_ENTRY(gtk_builder_get_object(builder, "entry_file_dest"));
	main_window_sub_widget.entry_file_dest = entry;

	g_signal_connect(window, "destroy",
			 G_CALLBACK (gtk_main_quit), &window);

	g_object_unref(G_OBJECT(builder));

	gtk_widget_show_all(window);
	gtk_main();
	return 0;
}
