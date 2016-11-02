package helloworld

import javafx.application.Application
import javafx.scene.Scene
import javafx.scene.control.Button
import javafx.scene.layout.StackPane
import javafx.stage.Stage

import groovy.transform.CompileStatic

@CompileStatic
class HelloWorldFx extends Application {

    static void main(String[] args) {
        launch(args)
    }

    @Override
    void start(Stage primaryStage) {

        def btn = new Button()

        btn.text     = "Say 'Hello World'"
        btn.onAction = { println 'Hello World!' }

        def root = new StackPane()

        root.children.add btn

        primaryStage.title = 'Hello World!'
        primaryStage.scene = new Scene(root, 300, 250)
        primaryStage.show()
    }
}

