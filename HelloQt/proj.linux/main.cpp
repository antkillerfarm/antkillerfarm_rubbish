#if 1
#include "../Classes/AppDelegate.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include "mainwindow.h"
#include <QApplication>

USING_NS_CC;

static pthread_t thread;

static void* thread_func(void* data)
{
    Application::getInstance()->run();

    QApplication::exit(0);
    return NULL;
}

int main(int argc, char **argv)
{
    // create the application instance
    int rc;
    QApplication a(argc, argv);
    MainWindow w;
    AppDelegate app;
    w.setCocosAppDelegate(&app);
    w.initGLWidget();
    w.show();

    pthread_create(&thread, NULL, thread_func, NULL);
    rc = a.exec();
    pthread_join(thread, NULL);

    return rc;
}
#else
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
#endif
