#include "CCQGLWidget.h"

NS_CC_BEGIN

CCQGLWidget::CCQGLWidget(int width, int height)
    : QGLWidget(QGLFormat(QGL::DoubleBuffer))
    , mouseMoveFunc(nullptr)
    , mousePressFunc(nullptr)
    , mouseReleaseFunc(nullptr)
    , wheelFunc(nullptr)
    , keyEventFunc(nullptr)
{
  ////////////////////////FOR - opengl error :missing opengl version /////////////////////////////////////
  QGLFormat format;
  //format.setVersion(QGLFormat::OpenGL_ES_Version_2_0, QGLFormat::OpenGL_ES_Version_2_0);
  format.setVersion(1, 1);
  format.setDoubleBuffer(true);
  ///////////////////////////////////////////////////////////////////////////
  setFormat(format);
  resize(width, height);
}

CCQGLWidget::~CCQGLWidget()
{
}

void CCQGLWidget::setMouseMoveFunc(MOUSE_PTRFUN func)
{
    mouseMoveFunc = func;
}

void CCQGLWidget::setMousePressFunc(MOUSE_PTRFUN func)
{
    mousePressFunc = func;
}

void CCQGLWidget::setMouseReleaseFunc(MOUSE_PTRFUN func)
{
    mouseReleaseFunc = func;
}

void CCQGLWidget::setWheelFunc(WHEEL_PTRFUN func)
{
	wheelFunc = func;
}

void CCQGLWidget::setKeyEventFunc(KEY_PTRFUN func)
{
    keyEventFunc = func;
}

void CCQGLWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (mouseMoveFunc)
        mouseMoveFunc(event);

    QGLWidget::mouseMoveEvent(event);
}

void CCQGLWidget::mousePressEvent(QMouseEvent *event)
{
    if (mousePressFunc)
        mousePressFunc(event);

    QGLWidget::mousePressEvent(event);
}

void CCQGLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (mouseReleaseFunc)
        mouseReleaseFunc(event);

    QGLWidget::mouseReleaseEvent(event);
}

void CCQGLWidget::wheelEvent(QWheelEvent *event)
{
	if (wheelFunc)
		wheelFunc(event);

	QGLWidget::wheelEvent(event);
}

void CCQGLWidget::keyPressEvent(QKeyEvent *event)
{
    if (keyEventFunc)
		keyEventFunc(event);

	QGLWidget::keyPressEvent(event);
}

void CCQGLWidget::keyReleaseEvent(QKeyEvent *event)
{
    if (keyEventFunc)
		keyEventFunc(event);

	QGLWidget::keyReleaseEvent(event);
}

NS_CC_END
