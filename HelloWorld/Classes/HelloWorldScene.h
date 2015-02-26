#ifndef __HELLOWORLD_SCENE_H__
#define __HELLOWORLD_SCENE_H__

#include "cocos2d.h"
class HelloWorld : public cocos2d::LayerColor
{
public:
    // there's no 'id' in cpp, so we recommend returning the class instance pointer
    static cocos2d::Scene* createScene();

    // Here's a difference. Method 'init' in cocos2d-x returns bool, instead of returning 'id' in cocos2d-iphone
    virtual bool init();
    
    // a selector callback
    void menuCloseCallback(cocos2d::Ref* pSender);
    void menuAnimationCallback(cocos2d::Ref* pSender);
    void duelistAnimationFinished();
    void horsemanAnimationFinished();
    
    // implement the "static create()" method manually
    CREATE_FUNC(HelloWorld);
    void myUpdate(float dt);

 private:
    cocos2d::Vec2 pos_duelist;
    cocos2d::Vec2 pos_horseman;
    cocos2d::Sprite*  sprite_duelist;
    cocos2d::Sprite*  sprite_horseman;
};

#endif // __HELLOWORLD_SCENE_H__
