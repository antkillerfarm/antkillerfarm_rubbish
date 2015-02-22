//#define COCOS2D_DEBUG 1
#include "HelloWorldScene.h"

USING_NS_CC;

Scene* HelloWorld::createScene()
{
    // 'scene' is an autorelease object
    auto scene = Scene::create();
    
    // 'layer' is an autorelease object
    auto layer = HelloWorld::create();

    // add layer as a child to scene
    scene->addChild(layer);

    // return the scene
    return scene;
}

// on "init" you need to initialize your instance
bool HelloWorld::init()
{
    //////////////////////////////
    // 1. super init first
    if ( !LayerColor::initWithColor(Color4B(155, 155, 155, 255)))
    {
        return false;
    }
    
    Size visibleSize = Director::getInstance()->getVisibleSize();
    Vec2 origin = Director::getInstance()->getVisibleOrigin();
    CCLOG("Visible Size: %f,%f", visibleSize.width, visibleSize.height);
    CCLOG("origin: %f,%f", origin.x, origin.y);


    /////////////////////////////
    // 2. add a menu item with "X" image, which is clicked to quit the program
    //    you may modify it.

    // add a "close" icon to exit the progress. it's an autorelease object
    auto closeItem = MenuItemImage::create(
                                           "img/CloseNormal.png",
                                           "img/CloseSelected.png",
                                           CC_CALLBACK_1(HelloWorld::menuCloseCallback, this));
    CCLOG("closeItem: %f,%f", closeItem->getContentSize().width, closeItem->getContentSize().height);

    auto vec = Vec2(origin.x + visibleSize.width - closeItem->getContentSize().width/2 ,
		    origin.y + visibleSize.height / 2 + closeItem->getContentSize().height/2);
    closeItem->setPosition(vec);

    auto closeItem1 = MenuItemImage::create(
					    "img/CloseSelected.png",
                                           "img/CloseNormal.png",
					    CC_CALLBACK_1(HelloWorld::menuAnimationCallback, this));

    vec = Vec2(origin.x + visibleSize.width - closeItem->getContentSize().width/2 ,
		    origin.y + visibleSize.height / 2 - closeItem->getContentSize().height/2);
    closeItem1->setPosition(vec);
    CCLOG("closeItem pos: %f,%f", vec.x, vec.y);

    // create menu, it's an autorelease object
    auto menu = Menu::create(closeItem, NULL);
    menu->setPosition(0, 60);
    this->addChild(menu, 1);
    menu = Menu::create(closeItem1, NULL);
    menu->setPosition(0, -60);
    this->addChild(menu, 2);

    /////////////////////////////
    // 3. add your codes below...

    // add a label shows "Hello World"
    // create and initialize a label
    
    auto label = Label::createWithTTF("Hello Antkillerfarm Studio!", "fonts/Marker Felt.ttf", 24);
    
    // position the label on the center of the screen
    label->setPosition(Vec2(origin.x + visibleSize.width/2,
                            origin.y + visibleSize.height - label->getContentSize().height));

    // add the label as a child to this layer
    this->addChild(label, 1);

    // add "HelloWorld" splash screen"
    auto sprite = Sprite::create("img/HelloWorld.png");

    // position the sprite on the center of the screen
    sprite->setPosition(Vec2(visibleSize.width/2 + origin.x, visibleSize.height/2 + origin.y));

    // add the sprite as a child to this layer
    this->addChild(sprite, 0);

    auto texture = Director::getInstance()->getTextureCache()->addImage("img/duelist.png");
    auto sprite1 = Sprite::createWithTexture(texture);
    auto size1 = sprite1->getContentSize();
    CCLOG("sprite1 size: %f,%f", size1.width, size1.height);

    sprite1->setAnchorPoint(Vec2(0.5, 0));
    sprite1->setPosition(Vec2(visibleSize.width/2 + origin.x, origin.y));
    this->addChild(sprite1, 0);

    auto animation = Animation::create();
    for( int i = 1; i < 9; i++)
    {
        char szName[100] = {0};
        sprintf(szName, "img/duelist-die%d.png", i);
        animation->addSpriteFrameWithFile(szName);
    }

    animation->setDelayPerUnit(1.0f / 9.0f);
    animation->setRestoreOriginalFrame(true);

    auto action = Animate::create(animation);
    sprite1->runAction(RepeatForever::create(Sequence::create(action, action->reverse(), nullptr)));

    sprite_duelist = Sprite::createWithTexture(texture);
    CCLOG("Addr: %x,%x", sprite1, sprite_duelist);
    sprite_duelist->setAnchorPoint(Vec2(0.5, 0));
    sprite_duelist->setPosition(Vec2(visibleSize.width/2 + origin.x, origin.y + size1.height));
    animation_duelist = animation->clone();
    //this->addChild(sprite_duelist, 0);

    //auto action1 = Animate::create(animation_duelist);
    //sprite_duelist->runAction(RepeatForever::create(Sequence::create(action1, action1->reverse(), nullptr)));

    return true;
}

void HelloWorld::menuAnimationCallback(Ref* pSender)
{
  this->addChild(sprite_duelist, 0);
  //auto action = Animate::create(animation_duelist);
  //sprite_duelist->runAction(Sequence::create(action, action->reverse(), nullptr));
}

void HelloWorld::menuCloseCallback(Ref* pSender)
{
#if (CC_TARGET_PLATFORM == CC_PLATFORM_WP8) || (CC_TARGET_PLATFORM == CC_PLATFORM_WINRT)
	MessageBox("You pressed the close button. Windows Store Apps do not implement a close button.","Alert");
    return;
#endif

    Director::getInstance()->end();

#if (CC_TARGET_PLATFORM == CC_PLATFORM_IOS)
    exit(0);
#endif
}
