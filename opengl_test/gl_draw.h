#ifndef GL_DRAW_H
#define GL_DRAW_H


void draw_a_sphere(unsigned int solid, double radius, int slices, int stacks);
void draw_a_circle(double radius, int slices);
void draw_a_rect();
void draw_a_test1();
void draw_rotate();
void draw_split();
void draw_a_wheel(gint index);
gboolean animation_timer_handler(gpointer user_data);
void draw_light_split();
void draw_sphere();
void draw_wheel();

#endif /* GL_DRAW_H */
