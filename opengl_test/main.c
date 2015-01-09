#include "gtkgl.h"

#define DRAW_STH_NORMAL 0
#define DRAW_STH_RECT 1
#define DRAW_STH_TEST1 2
#define DRAW_STH_ROTATE 3
#define DRAW_STH_SPLIT 4
#define DRAW_STH_WHEEL 5

gboolean init_flag = FALSE;
gint draw_sth_flag = DRAW_STH_NORMAL;

typedef struct{
	GtkWidget *main_window;
	GtkWidget *gl_window;
}MainWindowSubWidget;

MainWindowSubWidget main_window_sub_widget = {0};

static gint glwidget_draw (GtkWidget *widget, cairo_t *cr, gpointer userdata);

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

static void draw_a_sphere (unsigned int solid, double radius, int slices, int stacks)
{
	GLUquadricObj *quadObj = NULL;
	glColor3f(1.0, 1.0, 1.0);
	glLoadIdentity ();

	quadObj = gluNewQuadric ();

	if (solid)
		gluQuadricDrawStyle (quadObj, GLU_FILL);
	else
		gluQuadricDrawStyle (quadObj, GLU_LINE);

	gluQuadricNormals (quadObj, GLU_SMOOTH);
	gluSphere (quadObj, radius, slices, stacks);
}

gint rect_cnt = 0;
static void draw_a_rect ()
{
	gfloat z;
	glColor3f(1.0, 1.0, 1.0);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity ();

	gluPerspective(35, 1, 0.5, 20);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity ();
	z = 0.5 + rect_cnt * 0.5;
	gluLookAt (0, 0, z, 0, 0, 0, 0, 1, 0);
	rect_cnt++;
	rect_cnt = rect_cnt % 10;

	glBegin(GL_POLYGON);
	glVertex2f(-0.5,-0.5);
	glVertex2f(-0.5,0.5);
	glVertex2f(0.5,0.5);
	glVertex2f(0.5,-0.5);
	glEnd();

	glFlush();

	g_print("draw_a_rect:%f\r\n", z);
}

static void draw_a_test1()
{
	static GLfloat vertices[] = {-0.5, -0.5,
				     -0.5,0.5,
				     0.5,0.5,
				     0.5,-0.5};
	static GLint vertices1[] = {0, 0,
				   1, 1,
				    -1, 0,
				    0, 1};
	static GLfloat colors[] = {1.0, 0.2, 0.2,
				   0.2, 0.2, 1.0,
				   0.8, 1.0, 0.2,
				   0.75, 0.75, 0.75};

	glColor3f(1.0, 1.0, 1.0);
	glLoadIdentity ();

	glEnableClientState (GL_VERTEX_ARRAY);
	glEnableClientState (GL_COLOR_ARRAY);

	//glVertexPointer (2, GL_FLOAT, 0, vertices1);
	glVertexPointer (2, GL_INT, 0, vertices1);
	glColorPointer (3, GL_FLOAT, 0, colors);

	glBegin (GL_TRIANGLES);
	glArrayElement (0);
	glArrayElement (1);
	glArrayElement (2);
	glEnd ();

	glFlush();

	glDisableClientState (GL_VERTEX_ARRAY);
	glDisableClientState (GL_COLOR_ARRAY);
}

static void draw_rotate()
{
	GLfloat vertices[] = {-0.5, -0.5,
				     -0.5,0.5,
				     0.5,0.5,
				     0.5,-0.5};
	GLfloat colors[] = {1.0, 0.2, 0.2,
				   0.2, 0.2, 1.0,
				   0.8, 1.0, 0.2,
				   0.75, 0.75, 0.75};
	GLuint indices[4] = {0, 1, 2, 3};

	glColor3f(1.0, 1.0, 1.0);
	glLoadIdentity ();

	glEnableClientState (GL_VERTEX_ARRAY);
	glEnableClientState (GL_COLOR_ARRAY);

	glVertexPointer (2, GL_FLOAT, 0, vertices);
	glColorPointer (3, GL_FLOAT, 0, colors);

	glRotatef (45.0, 0.0, 0.0, 1.0);

	glDrawElements (GL_POLYGON, 4, GL_UNSIGNED_INT, indices);

	glFlush();

	glDisableClientState (GL_VERTEX_ARRAY);
	glDisableClientState (GL_COLOR_ARRAY);
}

static void draw_split()
{
	GtkAllocation alc;
	gtk_widget_get_allocation (main_window_sub_widget.gl_window, &alc);

	glColor3f(1.0, 1.0, 1.0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity ();

	gluPerspective(1, 1, 2, 20);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity ();
	gluLookAt (0, 0, -5, 0, 0, 0, 0, 1, 0);

	glViewport (0, 0, alc.width / 2, alc.height / 2);
	draw_rotate();

	glViewport (alc.width / 2, 0, alc.width / 2, alc.height / 2);
	draw_a_test1();

	glViewport (0, alc.height / 2, alc.width / 2, alc.height / 2);
	draw_a_rect();

	glViewport (alc.width / 2, alc.height / 2, alc.width / 2, alc.height / 2);
	draw_a_sphere (1, 0.5f, 100, 100);
}

static void draw_wheel()
{
	glColor3f(1.0, 1.0, 1.0);
	glLoadIdentity ();
}

static void opengl_scene_display (void)
{
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
	case DRAW_STH_WHEEL:  draw_wheel(); break;
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

static gint glwidget_draw (GtkWidget *widget, cairo_t *cr, gpointer userdata)
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
