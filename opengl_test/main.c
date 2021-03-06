#include <math.h>
#include "gtkgl.h"
#include "gl_draw.h"
#include "app.h"

gboolean init_flag = FALSE;

MainWindowSubWidget main_window_sub_widget = {0};
gint animation_flag = FALSE;
gint animation_index = 0;
DrawSthItem *draw_item_active = NULL;
DrawSthSubItem *draw_sub_active = NULL;
DrawCB draw_cb_active = NULL;
GLfloat adj_value[3] = {0, 0, 0};

#define DRAW_STH_NUM 2
#define DRAW_SHAPE_NUM 6
#define DRAW_LIGHT_NUM 3

DrawSthSubItem draw_shape[DRAW_SHAPE_NUM] = 
{
	0x0, "Rect", draw_a_rect, FALSE,
	0x1, "Sphere", draw_sphere, FALSE,
	0x2, "Rotate", draw_rotate, FALSE,
	0x3, "Test1", draw_a_test1, FALSE,
	0x4, "Split", draw_split, FALSE,
	0x5, "Wheel", draw_wheel, TRUE
};

DrawSthSubItem draw_light[DRAW_LIGHT_NUM] = 
{
	0x10, "Split", draw_light_split, FALSE,
	0x11, "Material", draw_light_sth, FALSE,
	0x12, "Light 2", draw_light_sth, FALSE
};

DrawSthItem draw_sth_item[DRAW_STH_NUM] = 
{
	"Shape", DRAW_SHAPE_NUM, draw_shape,
	"Light", DRAW_LIGHT_NUM, draw_light
};

DrawSthData draw_sth_data = 
{
	DRAW_STH_NUM, draw_sth_item
};

void update_cb_draw_type(gint index)
{
	gint i;
	GtkTreeIter iter;
	DrawSthItem *item = &(draw_sth_data.data[index]);
	
	gtk_list_store_clear(main_window_sub_widget.liststore_sub);
	for (i = 0; i < item->num; i++)
	{
		gtk_list_store_append(main_window_sub_widget.liststore_sub, &iter);
		gtk_list_store_set(main_window_sub_widget.liststore_sub, &iter, 0, item->sub_data[i].name, -1);
	}

	draw_item_active = item;
	draw_sub_active = &(draw_item_active->sub_data[0]);
	draw_cb_active  = draw_sub_active->draw_cb;
	gtk_combo_box_set_active(main_window_sub_widget.cb_draw_sub, 0);
}

G_MODULE_EXPORT void cb_draw_type_changed(GtkComboBox *widget, gpointer user_data)
{
	gint index = gtk_combo_box_get_active(widget);
	update_cb_draw_type(index);
	animation_flag = FALSE;
	//gdouble value = gtk_adjustment_get_value(main_window_sub_widget.adjustment[0]);
	//g_print("v:%f\r\n", value);
}

G_MODULE_EXPORT void cb_draw_sub_changed(GtkComboBox *widget, gpointer user_data)
{
	gint index = gtk_combo_box_get_active(widget);
	draw_sub_active = &(draw_item_active->sub_data[index]);
	draw_cb_active  = draw_sub_active->draw_cb;
	animation_flag = FALSE;
}

void refresh_scale_value()
{
	adj_value[0] = gtk_adjustment_get_value(main_window_sub_widget.adjustment[0]);
	adj_value[1] = gtk_adjustment_get_value(main_window_sub_widget.adjustment[1]);
	adj_value[2] = gtk_adjustment_get_value(main_window_sub_widget.adjustment[2]);
	gtk_widget_queue_draw(main_window_sub_widget.gl_window);
}

G_MODULE_EXPORT void do_btn_cb_draw(GtkButton *button, gpointer data)
{
	if (draw_sub_active->is_animation == TRUE)
	{
		if (animation_flag)
		{
			animation_flag = FALSE;
		}
		else
		{
			animation_flag = TRUE;
			g_timeout_add(50, animation_timer_handler, NULL);
		}
	}
	else
	{
		animation_flag = TRUE;
	}
	refresh_scale_value();
}

G_MODULE_EXPORT void do_scale_changed(GtkRange *range, gpointer  user_data)
{
	refresh_scale_value();
}

static void opengl_scene_init (void)
{

}

static void opengl_scene_configure (void)
{
        /* 设置投影矩阵 */
        //glMatrixMode (GL_PROJECTION);
        //glLoadIdentity ();
        //glOrtho (-1., 1., -1., 1., -1., 20.);

        /* 设置模型视图 */
        //glMatrixMode (GL_MODELVIEW);
        //glLoadIdentity ();
        //glTranslatef (0., 0., -10.);
//        gluLookAt (0., 0., 10., 0., 0., 0., 0., 1., 0.);
}

static void opengl_scene_display (void)
{
	if (animation_flag == FALSE)
	{
		return;
	}
	GtkAllocation alc;
	gtk_widget_get_allocation (main_window_sub_widget.gl_window, &alc);

        /* 背景 */
	glClearColor (0.2, 0.4, 0.6, 1.0);
	glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport (0, 0, alc.width, alc.height);

        /* 绘制几何体 */
	draw_cb_active();
}

static void glwidget_show (GtkWidget *widget, gpointer userdata)
{
	gtk_gl_enable (widget);
	gtk_gl_make_current (widget);

	opengl_scene_init ();
	init_flag = TRUE;
	//g_print("glwidget_show\r\n");
}

static gboolean glwidget_configure (GtkWidget *widget, GdkEventConfigure *event, gpointer userdata)
{       
	GtkAllocation alc;
	//g_print("glwidget_configure\r\n");
	if (init_flag == FALSE)
	{
		glwidget_show(widget, NULL);
	}
	gtk_widget_get_allocation (widget, &alc);
       
	gtk_gl_make_current (widget);

	glViewport (0, 0, alc.width, alc.height);
	opengl_scene_configure ();
        
	return TRUE;
}

gint glwidget_draw (GtkWidget *widget, cairo_t *cr, gpointer userdata)
{       
	gtk_gl_make_current (widget);

	opengl_scene_display ();

	gtk_gl_swap_buffers (widget);
 
	return TRUE;
}

static void glwidget_destory (GtkWidget *widget,  gpointer userdata)
{
        gtk_gl_disable (widget);
}

void init_cb_draw()
{
	gint i;
	GtkTreeIter iter;
	for (i = 0; i < draw_sth_data.num; i++)
	{
		gtk_list_store_append(main_window_sub_widget.liststore_draw, &iter);
		gtk_list_store_set(main_window_sub_widget.liststore_draw, &iter, 0, draw_sth_data.data[i].name, -1);
	}
	gtk_combo_box_set_active(main_window_sub_widget.cb_draw_type, 0);
	update_cb_draw_type(0);
}

int main (int argc, char **argv)
{
	GError *err = NULL;
	GtkBuilder *builder;

	gtk_init (&argc, &argv);
 
	builder = gtk_builder_new();
	gtk_builder_add_from_file(builder, "demo.glade", &err);
	gtk_builder_connect_signals(builder, NULL);

	main_window_sub_widget.main_window = GTK_WIDGET(gtk_builder_get_object(builder, "main_window"));
	gtk_window_set_title (GTK_WINDOW (main_window_sub_widget.main_window), "The OpenGL support of GTK+ 3.0");
	g_signal_connect (main_window_sub_widget.main_window, "destroy", G_CALLBACK (gtk_main_quit), NULL);

	main_window_sub_widget.gl_window = GTK_WIDGET(gtk_builder_get_object(builder, "drawingarea1"));
	g_signal_connect (main_window_sub_widget.gl_window, "configure-event", G_CALLBACK (glwidget_configure), NULL);
	g_signal_connect (main_window_sub_widget.gl_window, "draw", G_CALLBACK (glwidget_draw), NULL);
	g_signal_connect (main_window_sub_widget.gl_window, "destroy", G_CALLBACK (glwidget_destory), NULL);

	main_window_sub_widget.adjustment[0] = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adjustment0"));
	main_window_sub_widget.adjustment[1] = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adjustment1"));
	main_window_sub_widget.adjustment[2] = GTK_ADJUSTMENT(gtk_builder_get_object(builder, "adjustment2"));

	main_window_sub_widget.cb_draw_type = GTK_COMBO_BOX(gtk_builder_get_object(builder, "cb_draw_type"));
	main_window_sub_widget.cb_draw_sub = GTK_COMBO_BOX(gtk_builder_get_object(builder, "cb_draw_sub"));

	main_window_sub_widget.liststore_draw = GTK_LIST_STORE(gtk_builder_get_object(builder, "liststore_draw"));
	main_window_sub_widget.liststore_sub = GTK_LIST_STORE(gtk_builder_get_object(builder, "liststore_sub"));

	init_cb_draw();

	g_object_unref(G_OBJECT(builder));
	gtk_widget_show_all(main_window_sub_widget.main_window);

	gtk_main ();
        
	return 0;
}
