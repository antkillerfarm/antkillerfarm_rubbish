#include <math.h>
#include "gtkgl.h"
#include "gl_draw.h"
#include "app.h"

#define DRAW_STH_NORMAL 0
#define DRAW_STH_RECT 1
#define DRAW_STH_TEST1 2
#define DRAW_STH_ROTATE 3
#define DRAW_STH_SPLIT 4
#define DRAW_STH_WHEEL 5
#define DRAW_STH_LIGHT 6

gboolean init_flag = FALSE;
gint draw_sth_flag = DRAW_STH_NORMAL;

MainWindowSubWidget main_window_sub_widget = {0};
gint animation_flag = FALSE;
gint animation_index = 0;

G_MODULE_EXPORT void do_btn_draw_rect(GtkButton *button, gpointer data)
{
	draw_sth_flag = DRAW_STH_RECT;
	glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
}

G_MODULE_EXPORT void do_btn_draw_test1(GtkButton *button, gpointer data)
{
	draw_sth_flag = DRAW_STH_TEST1;
	glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
}

G_MODULE_EXPORT void do_btn_draw_rotate(GtkButton *button, gpointer data)
{
	draw_sth_flag = DRAW_STH_ROTATE;
	glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
}

G_MODULE_EXPORT void do_btn_draw_split(GtkButton *button, gpointer data)
{
	draw_sth_flag = DRAW_STH_SPLIT;
	glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
}

G_MODULE_EXPORT void do_btn_draw_wheel(GtkButton *button, gpointer data)
{
	draw_sth_flag = DRAW_STH_WHEEL;
	if (animation_flag)
	{
		animation_flag = FALSE;
	}
	else
	{
		animation_flag = TRUE;
		g_timeout_add(500, animation_timer_handler, NULL);
	}
}

G_MODULE_EXPORT void do_btn_draw_light(GtkButton *button, gpointer data)
{
	draw_sth_flag = DRAW_STH_LIGHT;
	glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
}

G_MODULE_EXPORT void cb_draw_type_changed(GtkComboBox *widget, gpointer user_data)
{
	draw_sth_flag = gtk_combo_box_get_active(widget);
	animation_flag = FALSE;
}

G_MODULE_EXPORT void do_btn_cb_draw(GtkButton *button, gpointer data)
{
	if (draw_sth_flag == DRAW_STH_WHEEL)
	{
		if (animation_flag)
		{
			animation_flag = FALSE;
		}
		else
		{
			animation_flag = TRUE;
			g_timeout_add(500, animation_timer_handler, NULL);
		}
	}
	else
	{
		animation_flag = TRUE;
	}
	glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
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
	switch(draw_sth_flag)
	{
	case DRAW_STH_NORMAL:  draw_a_sphere (1, 0.5f, 100, 100); break;
	case DRAW_STH_RECT:  draw_a_rect(); break;
	case DRAW_STH_TEST1:  draw_a_test1(); break;
	case DRAW_STH_ROTATE:  draw_rotate(); break;
	case DRAW_STH_SPLIT:  draw_split(); break;
	case DRAW_STH_WHEEL:  draw_wheel(animation_index); break;
	case DRAW_STH_LIGHT:  draw_light(); break;
	}
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

	g_object_unref(G_OBJECT(builder));
	gtk_widget_show_all(main_window_sub_widget.main_window);

	gtk_main ();
        
	return 0;
}
