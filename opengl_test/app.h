#ifndef APP_H
#define APP_H

typedef struct{
	GtkWidget *main_window;
	GtkWidget *gl_window;
}MainWindowSubWidget;

extern MainWindowSubWidget main_window_sub_widget;
extern gint animation_flag;
extern gint animation_index;

gint glwidget_draw (GtkWidget *widget, cairo_t *cr, gpointer userdata);

#endif /* APP_H */
