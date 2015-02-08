#ifndef APP_H
#define APP_H

typedef struct{
	GtkWidget *main_window;
	GtkWidget *gl_window;
	GtkAdjustment *adjustment[3];
	GtkListStore *liststore_draw;
	GtkListStore *liststore_sub;
	GtkComboBox *cb_draw_type;
	GtkComboBox *cb_draw_sub;
}MainWindowSubWidget;

typedef void (*DrawCB)(void);

typedef struct{
	guint id;
	gchararray name;
	DrawCB draw_cb;
	gboolean is_animation;
}DrawSthSubItem;

typedef struct{
	gchararray name;
	gint num;
	DrawSthSubItem *sub_data;
}DrawSthItem;

typedef struct{
	gint num;
	DrawSthItem *data;
}DrawSthData;

extern MainWindowSubWidget main_window_sub_widget;
extern gint animation_flag;
extern gint animation_index;
extern GLfloat adj_value[3];
extern DrawSthSubItem *draw_sub_active;

gint glwidget_draw (GtkWidget *widget, cairo_t *cr, gpointer userdata);

#endif /* APP_H */
