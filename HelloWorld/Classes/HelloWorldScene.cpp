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

void HelloWorld::myUpdate(float dt)  
{  
  //CCLOG("upadte----%f",dt);  
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

    auto texture1 = Director::getInstance()->getTextureCache()->addImage("img/duelist.png");
    auto sprite1 = Sprite::createWithTexture(texture1);
    auto size1 = sprite1->getContentSize();
    CCLOG("sprite1 size: %f,%f", size1.width, size1.height);

    sprite1->setAnchorPoint(Vec2(0.5, 0));
    sprite1->setPosition(Vec2(visibleSize.width/2 + origin.x, origin.y));
    this->addChild(sprite1, 0);

    auto animation1 = Animation::create();
    for( int i = 1; i < 9; i++)
    {
        char szName[100] = {0};
        sprintf(szName, "img/duelist-die%d.png", i);
        animation1->addSpriteFrameWithFile(szName);
    }

    animation1->setDelayPerUnit(1.0f / 9.0f);
    animation1->setRestoreOriginalFrame(true);
    AnimationCache::getInstance()->addAnimation (animation1, "duelist-die");

    auto action1 = Animate::create(animation1);
    sprite1->runAction(RepeatForever::create(Sequence::create(action1, action1->reverse(), nullptr)));

    pos_duelist.set(visibleSize.width / 2 + origin.x, origin.y + size1.height);
    pos_horseman.set(visibleSize.width / 2 - 200 + origin.x, origin.y + size1.height);
    sprite_duelist = nullptr;

    auto texture2 = Director::getInstance()->getTextureCache()->addImage("img/horseman-se.png");
    sprite_horseman = Sprite::createWithTexture(texture2);

    sprite_horseman->setAnchorPoint(Vec2(0.5, 0));
    //sprite_horseman->setPosition(Vec2(size1.width + origin.x, origin.y + size1.height));
    sprite_horseman->setVisible(false);
    sprite_horseman->setScale(2.0);
    this->addChild(sprite_horseman, 1);

   auto animation2 = Animation::create();
    for( int i = 1; i < 13; i++)
    {
        char szName[100] = {0};
        sprintf(szName, "img/horseman-se-attack%d.png", i);
        animation2->addSpriteFrameWithFile(szName);
    }

    animation2->setDelayPerUnit(1.2f / 12.0f);
    animation2->setRestoreOriginalFrame(true);
    AnimationCache::getInstance()->addAnimation (animation2, "horseman-se-attack");

    auto texture = Director::getInstance()->getTextureCache()->addImage("img/duelist.png");
    sprite_duelist = Sprite::createWithTexture(texture);
    sprite_duelist->setAnchorPoint(Vec2(0.5, 0));
    sprite_duelist->setPosition(pos_duelist);
    sprite_duelist->setScale(2.0);
    this->addChild(sprite_duelist, 0);

    //scheduleUpdate();
    //schedule(schedule_selector(HelloWorld::myUpdate),1.0f);

    auto texture3 = Director::getInstance()->getTextureCache()->addImage("img/windmill_white.png");
    auto sprite_windmill = Sprite::createWithTexture(texture3);
    auto size_windmill = sprite1->getContentSize();
    sprite_windmill->setAnchorPoint(Vec2(0.5078, 0.3594));
    sprite_windmill->setPosition(Vec2(size_windmill.width * 2.5 + origin.x, visibleSize.height / 2 + origin.y));
    auto rotateBy = RotateBy::create(2.0f, 360.0f);
    sprite_windmill->runAction(RepeatForever::create(rotateBy));
    //sprite_windmill->setOpacity(30);
    sprite_windmill->setColor(Color3B::BLUE);
    this->addChild(sprite_windmill, 0);

    return true;
}

void HelloWorld::menuAnimationCallback(Ref* pSender)
{
  auto texture = Director::getInstance()->getTextureCache()->addImage("img/duelist.png");
  sprite_duelist->setTexture(texture);

  sprite_horseman->setPosition(pos_horseman);
  sprite_horseman->setVisible(true);
  auto animation_horseman = AnimationCache::getInstance()->getAnimation("horseman-se-attack");
  auto action2 = Animate::create(animation_horseman);
  auto walkLeft = MoveBy::create(1.2, Vec2(400,0));
  auto seq = Spawn::create(action2, walkLeft, nullptr);
  sprite_horseman->runAction(Sequence::create(seq, CallFunc::create(CC_CALLBACK_0(HelloWorld::horsemanAnimationFinished, this)), nullptr)); 
}

void HelloWorld::horsemanAnimationFinished()
{
  auto animation_duelist = AnimationCache::getInstance()->getAnimation("duelist-die");
  auto action = Animate::create(animation_duelist);
  sprite_duelist->runAction(Sequence::create(action, CallFunc::create(CC_CALLBACK_0(HelloWorld::duelistAnimationFinished, this)), nullptr));
}

void HelloWorld::duelistAnimationFinished()
{
  auto texture = Director::getInstance()->getTextureCache()->addImage("img/duelist-die8.png");
  sprite_duelist->setTexture(texture);
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
