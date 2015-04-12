#include "HelloWorldScene.h"
#include "audio/include/SimpleAudioEngine.h"

#define PTM_RATIO 32
#define EMIT_NUM 11


USING_NS_CC;
using namespace CocosDenshion;

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
    if ( !Layer::init() )
    {
        return false;
    }
    
    auto audio = SimpleAudioEngine::getInstance(); 
    // set the background music and continuously play it. 
    audio->playBackgroundMusic("sanguo.mp3", true); 

    Size visibleSize = Director::getInstance()->getVisibleSize();
    Vec2 origin = Director::getInstance()->getVisibleOrigin();

    auto label1 = Label::createWithTTF("滚滚长江东逝水", "fonts/Fat_GBK.ttf", 24);

    label1->setPosition(Vec2(origin.x + label1->getContentSize().width / 2,
                            origin.y + visibleSize.height - label1->getContentSize().height));

    auto walk_right = MoveBy::create(6, Vec2(visibleSize.width - label1->getContentSize().width, 0));
    label1->runAction(RepeatForever::create(Sequence::create(walk_right, walk_right->reverse(), nullptr)));

    // add the label as a child to this layer
    this->addChild(label1, 1);

    auto label2 = Label::createWithTTF("浪花淘尽英雄", "fonts/Fat_GBK.ttf", 24);

    label2->setPosition(Vec2(origin.x + visibleSize.width - label2->getContentSize().width / 2,
                            origin.y + visibleSize.height - 3 * label2->getContentSize().height));

    auto walk_left = MoveBy::create(6, Vec2(-(visibleSize.width - label2->getContentSize().width), 0));
    label2->runAction(RepeatForever::create(Sequence::create(walk_left, walk_left->reverse(), nullptr)));

    // add the label as a child to this layer
    this->addChild(label2, 1);

    emitter = ParticleSun::create();
    emitter->setPosition(Vec2(origin.x + visibleSize.width / 2,
			      origin.y + visibleSize.height / 2));
    this->addChild(emitter, 10);

    //MYCode
    Size winSize = Director::getInstance()->getWinSize();
    b2Vec2 gravity;
    gravity.Set(5.0f,-10.0f);
    world = new b2World(gravity);
    //b2Body* groundBody = 
    createGround(0,0,winSize.width,winSize.height);
    //this->setTouchEnabled(true);
    //this->setAccelerometerEnabled(true);
    //this->setAccelerometerEnabled(true);

    auto listener1 = EventListenerTouchOneByOne::create();
 
    // trigger when you push down
    listener1->onTouchBegan = [=](Touch* touch, Event* event){
      //Touch*touch = (Touch*)pTouches->anyObject();
      Point location = touch->getLocation();
      createSprite(location, (char*)("football.png"));
      return true; // if you are consuming it
    };
 
    // trigger when moving touch
    listener1->onTouchMoved = [](Touch* touch, Event* event){
      // your code
    };
 
    // trigger when you let up
    listener1->onTouchEnded = [](Touch* touch, Event* event){
      // your code
    };
 
    // Add listener
    _eventDispatcher->addEventListenerWithSceneGraphPriority(listener1, this);

	update_cnt = 0;
	schedule( schedule_selector(HelloWorld::myupdate), 6.0);

    scheduleUpdate();
    
    return true;
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

b2Body* HelloWorld::createGround(int sx,int sy,int width,int height)
{
  b2BodyDef groundBodyDef;
  groundBodyDef.position.Set(sx,sy);
  b2Body* groundBody = world->CreateBody(&groundBodyDef);
  b2EdgeShape groundBox;
  //bottom
  groundBox.Set(b2Vec2(0,0), b2Vec2(width / PTM_RATIO, 0));
  groundBody->CreateFixture(&groundBox,0);
  // top
  groundBox.Set(b2Vec2(0, height / PTM_RATIO), b2Vec2(width / PTM_RATIO, height / PTM_RATIO));
  groundBody->CreateFixture(&groundBox,0);
  // left
  groundBox.Set(b2Vec2(0, height / PTM_RATIO), b2Vec2(0, 0));
  groundBody->CreateFixture(&groundBox, 0);
  // right
  groundBox.Set(b2Vec2(width / PTM_RATIO, height / PTM_RATIO), b2Vec2(width / PTM_RATIO, 0));
  groundBody->CreateFixture(&groundBox,0);
  return groundBody;
}

void HelloWorld::update(float dt)
{
  int velocityIterations = 8;
  int positionIterations = 1;
  world->Step(dt, velocityIterations, positionIterations);
  for(b2Body* b = world->GetBodyList(); b ; b = b->GetNext())
    {
      if(b->GetUserData() != NULL)
	{
	  Sprite* mActor = (Sprite*)b->GetUserData();
	  mActor->setPosition(Point(b->GetPosition().x*PTM_RATIO, b->GetPosition().y*PTM_RATIO));
	  mActor->setRotation(-1*CC_RADIANS_TO_DEGREES(b->GetAngle()));//note
	}
    }
}

void HelloWorld::createSprite(Point location, char image[])
{
  Sprite* sprite = Sprite::create(image);
  this->addChild(sprite);
  sprite->setPosition(location);
  b2BodyDef bodyDef;
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(location.x/PTM_RATIO,location.y/PTM_RATIO);
  bodyDef.userData = sprite;
  b2Body* body = world->CreateBody(&bodyDef);
  b2CircleShape dynamicBox;
  dynamicBox.m_radius =12.0f/PTM_RATIO;
  b2FixtureDef fixtureDef;
  fixtureDef.shape = &dynamicBox;
  fixtureDef.density = 1.0f;
  fixtureDef.friction = 0.3f;
  fixtureDef.restitution = 0.8f;
  body->CreateFixture(&fixtureDef);
}

/*void HelloWorld::onTouchEnded(Touch* touch, Event* unused_event)
{
  //Touch*touch = (Touch*)pTouches->anyObject();
  Point location = touch->getLocation();
  createSprite(location,"football.png");
}
*/

void HelloWorld::myupdate(float dt)
{
  update_cnt++;
  update_cnt = update_cnt % EMIT_NUM;
  this->removeChild(emitter);
  switch (update_cnt)
    {
    case 0: emitter = ParticleSun::create();
      break;
    case 1: emitter = ParticleFlower::create();
      break;
    case 2: emitter = ParticleMeteor::create();
      break;
    case 3: emitter = ParticleExplosion::create();
      break;
    case 4: emitter = ParticleSmoke::create();
      break;
    case 5: emitter = ParticleSnow::create();
      break;
    case 6: emitter = ParticleRain::create();
      break;
    case 7: emitter = ParticleSpiral::create();
      break;
    case 8: emitter = ParticleFire::create();
      break;
    case 9: emitter = ParticleFireworks::create();
      break;
    case 10: emitter = ParticleGalaxy::create();
      break;
    }
  Size visibleSize = Director::getInstance()->getVisibleSize();
  Vec2 origin = Director::getInstance()->getVisibleOrigin();

  emitter->setPosition(Vec2(origin.x + visibleSize.width / 2,
			    origin.y + visibleSize.height / 2));
  this->addChild(emitter, 10);
}
