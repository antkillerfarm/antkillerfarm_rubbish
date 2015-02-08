#include <math.h>
#include "gtkgl.h"
#include "gl_draw.h"
#include "app.h"

void draw_a_sphere (unsigned int solid, double radius, int slices, int stacks)
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

void draw_sphere()
{
	draw_a_sphere (1, 0.5f, 100, 100);
}

#define PI 3.141592653589793
void draw_a_circle (double radius, int slices)
{
	GLfloat vertices[slices * 2];
	gint i;
	GLfloat angle;

	glEnableClientState (GL_VERTEX_ARRAY);
	for (i = 0; i < slices; i++)
	{
		angle = 2 * PI * i  / slices;
		vertices[i * 2] = radius * sin(angle);
		vertices[i * 2 + 1] = radius * cos(angle);
	}
	glVertexPointer (2, GL_FLOAT, 0, vertices);
	glDrawArrays(GL_POLYGON, 0, slices);
	glDisableClientState (GL_VERTEX_ARRAY);
}

gint rect_cnt = 0;
void draw_a_rect ()
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

void draw_a_test1()
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

void draw_rotate()
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

void draw_split()
{
	GtkAllocation alc;
	gtk_widget_get_allocation (main_window_sub_widget.gl_window, &alc);

	glColor3f(1.0, 1.0, 1.0);
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity ();

	//gluPerspective(1, 1, 2, 20);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity ();
	//gluLookAt (0, 0, -5, 0, 0, 0, 0, 1, 0);

	glViewport (0, 0, alc.width / 2, alc.height / 2);
	draw_rotate();

	glViewport (alc.width / 2, 0, alc.width / 2, alc.height / 2);
	draw_a_test1();

	//glViewport (0, alc.height / 2, alc.width / 2, alc.height / 2);
	//draw_a_rect();

	glViewport (alc.width / 2, alc.height / 2, alc.width / 2, alc.height / 2);
	draw_a_sphere (1, 0.5f, 100, 100);
}

void draw_a_wheel(gint index)
{
	gint i;
	for (i = 0; i < 8; i++)
	{
		glPushMatrix();
		glColor3f(0.125 * i, 0.125 * i, 0.0);
		glRotatef (45.0 * (i + index), 0.0, 0.0, 1.0);
		glTranslatef(0.5, 0 , 0);
		draw_a_circle(0.1, 36);
		glPopMatrix();
	}
}

void draw_wheel()
{
	draw_a_wheel(animation_index);
}

gboolean animation_timer_handler(gpointer user_data)
{
	if (animation_flag)
	{
		animation_index++;
		animation_index = animation_index % 8;
		glwidget_draw(main_window_sub_widget.gl_window, NULL, NULL);
		return TRUE;
	}
	else
	{
		return FALSE;
	}
}

void draw_light_split()
{
	GtkAllocation alc;
	gtk_widget_get_allocation (main_window_sub_widget.gl_window, &alc);

	GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_shininess[] = { 0.0 };
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };

	glViewport (0, 0, alc.width / 2, alc.height / 2);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho (-1.5, 1.5, -1.5, 
	    1.5, -10.0, 10.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity ();
	//glPushMatrix();
	//gluLookAt (0, 0, 5, 0, 0, 1, 0, 1, 0);
	//glPopMatrix();

	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//gluLookAt (0, 1, -1, 0, 0, 0, 0, 1, 1);

        /*glBegin(GL_POLYGON);
	glVertex2f(-0.5,-0.5);
	glVertex2f(-0.5,0.5);
	glVertex2f(0.5,0.5);
	glVertex2f(0.5,-0.5);
	glEnd();

	glFlush();*/
	draw_a_sphere (1, 0.7f, 100, 100);

	glViewport (alc.width / 2, 0, alc.width / 2, alc.height / 2);
	mat_shininess[0] = 30.0;
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	draw_a_sphere (1, 0.7f, 100, 100);

	glViewport (0, alc.height / 2, alc.width / 2, alc.height / 2);
	mat_shininess[0] = 70.0;
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	draw_a_sphere (1, 0.7f, 100, 100);

	glViewport (alc.width / 2, alc.height / 2, alc.width / 2, alc.height / 2);
	mat_shininess[0] = 100.0;
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	draw_a_sphere (1, 0.7f, 100, 100);
}

GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat mat_shininess[] = { 50.0 };
GLfloat mat_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
GLfloat mat_emission[] = { 0.0, 0.0, 0.0, 1.0 };
GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 1.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 1.0 };
GLfloat light2_position[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light2_specular[] = { 1.0, 0.0, 0.0, 0.0 };

void draw_light_sth()
{
	gint i;

	if (draw_sub_active->id == 0x11)
	{
	        mat_shininess[0] = adj_value[0];
		for (i = 0; i < 3; i++)
		{
			mat_ambient[i] = adj_value[1] * 0.01;
		}
		mat_emission[1] = adj_value[2] * 0.01;
	}
	else
	{
		light2_position[0] = adj_value[0] * 0.01 - 0.5;
		light2_position[1] = adj_value[1] * 0.01 - 0.5;
		light2_position[2] = adj_value[2] * 0.01 - 0.5;
	}

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho (-1.5, 1.5, -1.5, 
	    1.5, -10.0, 10.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity ();
	//glPushMatrix();
	//gluLookAt (0, 0, 5, 0, 0, 1, 0, 1, 0);
	//glPopMatrix();

	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT1, GL_POSITION, light2_position);
	glLightfv(GL_LIGHT1, GL_SPECULAR, light2_specular);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//gluLookAt (0, 1, -1, 0, 0, 0, 0, 1, 1);

        /*glBegin(GL_POLYGON);
	glVertex2f(-0.5,-0.5);
	glVertex2f(-0.5,0.5);
	glVertex2f(0.5,0.5);
	glVertex2f(0.5,-0.5);
	glEnd();

	glFlush();*/
	draw_a_sphere (1, 0.7f, 100, 100);
}
