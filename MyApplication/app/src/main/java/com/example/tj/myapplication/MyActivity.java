package com.example.tj.myapplication;

import android.app.Activity;
import android.app.Fragment;
import android.app.FragmentTransaction;
import android.os.Bundle;
import android.support.v4.view.PagerAdapter;
import android.support.v4.view.PagerTitleStrip;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.support.v4.view.ViewPager;
import android.support.v4.view.ViewPager.OnPageChangeListener;
import android.widget.RadioButton;

import java.util.ArrayList;
import java.util.List;


public class MyActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_my);
        if (savedInstanceState == null) {
            getFragmentManager().beginTransaction()
                    .add(R.id.container, new PlaceholderFragment())
                    .commit();
        }
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.my, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();
        if (id == R.id.action_settings) {
            Log.i("MyGUI", "action_settings");
            Log.i("MyGUI", "" + Runtime.getRuntime().maxMemory());
            showAnotherFragment();
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    public void showAnotherFragment() {
        FragmentTransaction transaction = getFragmentManager().beginTransaction();
        transaction.replace(R.id.container, new PlaceholderFragment2());
        transaction.addToBackStack(null);
        transaction.commit();
    }

    /**
     * A placeholder fragment containing a simple view.
     */
    public static class PlaceholderFragment extends Fragment {

        public PlaceholderFragment() {
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                Bundle savedInstanceState) {
            return inflater.inflate(R.layout.fragment_my, container, false);
        }
    }

    public static class PlaceholderFragment2 extends Fragment implements OnPageChangeListener {

        public PlaceholderFragment2() {
        }

        private ViewPager awesomePager;
        private AwesomePagerAdapter awesomeAdapter;

        private LayoutInflater mInflater;
        private List<View> mListViews;
        private List<String> mTitles;

        private class AwesomePagerAdapter extends PagerAdapter {

            @Override
            public int getCount() {
                return mListViews.size();
            }

            @Override
            public boolean isViewFromObject(View view, Object object) {
                return view == (object);
            }

            @Override
            public void destroyItem(ViewGroup container, int position, Object object) {
                Log.i("MyGUI", "destroyItem:" + position);
                container.removeView(mListViews.get(position));
            }

            @Override
            public Object instantiateItem(ViewGroup container, int position) {
                Log.i("MyGUI", "instantiateItem:" + position);
                View view = mListViews.get(position);
                container.addView(view, 0);
                return view;
            }

            public CharSequence getPageTitle(int position) {
                return mTitles.get(position);
            }
        }

        private static final int Radio_Depth = 0;
        private static final int Radio_Flip = 1;
        private static final int Radio_Default = 2;
        private int radio = Radio_Default;

        private void InitRadio(View view) {
            RadioButton rb;
            if (radio == Radio_Depth)
            {
                rb = (RadioButton)view.findViewById(R.id.rbtn_depth);
                awesomePager.setPageTransformer(true, new DepthPageTransformer());
            }
            else if (radio == Radio_Flip)
            {
                rb = (RadioButton)view.findViewById(R.id.rbtn_flip);
            }
            else
            {
                rb = (RadioButton)view.findViewById(R.id.rbtn_default);
                awesomePager.setPageTransformer(true, null);
            }
            rb.toggle();
        }

        @Override
        public View onCreateView(LayoutInflater inflater, ViewGroup container,
                                 Bundle savedInstanceState) {
            View view = inflater.inflate(R.layout.fragment_my2, container, false);

            mListViews = new ArrayList<View>();
            mInflater = getActivity().getLayoutInflater();
            mListViews.add(mInflater.inflate(R.layout.layout1, null));
            mListViews.add(mInflater.inflate(R.layout.layout0, null));
            mListViews.add(mInflater.inflate(R.layout.layout1, null));
            mListViews.add(mInflater.inflate(R.layout.layout0, null));

            awesomeAdapter = new AwesomePagerAdapter();
            awesomePager = (ViewPager)view.findViewById(R.id.viewpager);
            awesomePager.setAdapter(awesomeAdapter);
            awesomePager.setOnPageChangeListener(this);
            awesomePager.setCurrentItem(1);

            mTitles = new ArrayList<String>();
            mTitles.add("Rome");
            mTitles.add("Freedom");
            mTitles.add("Rome");
            mTitles.add("Freedom");


            RadioButton rb = (RadioButton)view.findViewById(R.id.rbtn_depth);
            rb.setOnClickListener(mButtonDepthListener);
            rb = (RadioButton)view.findViewById(R.id.rbtn_flip);
            rb.setOnClickListener(mButtonFlipListener);
            rb = (RadioButton)view.findViewById(R.id.rbtn_default);
            rb.setOnClickListener(mButtonDefaultListener);
            InitRadio(view);
            return view;
        }

        private View.OnClickListener mButtonDepthListener = new View.OnClickListener() {
            public void onClick(View v) {
                radio = Radio_Depth;
                awesomePager.setPageTransformer(true, new DepthPageTransformer());
                Log.i("MyGUI", "Radio_Depth");
            }
        };

        private View.OnClickListener mButtonFlipListener = new View.OnClickListener() {
            public void onClick(View v) {
                radio = Radio_Flip;
                awesomePager.setPageTransformer(true, new DepthSuperPageTransformer());
                Log.i("MyGUI", "Radio_Flip");
            }
        };

        private View.OnClickListener mButtonDefaultListener = new View.OnClickListener() {
            public void onClick(View v) {
                radio = Radio_Default;
                awesomePager.setPageTransformer(true, new DefaultPageTransformer());
                Log.i("MyGUI", "Radio_Default");
            }
        };

        @Override
        public void onPageSelected(int position) {
            Log.i("MyGUI", "onPageSelected:" + position);

            int pageIndex = position;
            if (position == 0) {
                pageIndex = mListViews.size() - 2;
            } else if (position == mListViews.size() - 1) {
                pageIndex = 1;
            }
            if (position != pageIndex) {
                awesomePager.setCurrentItem(pageIndex, false);
            }
        }

        @Override
        public void onPageScrollStateChanged(int state) {

        }

        @Override
        public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {

        }

        public class DepthPageTransformer implements ViewPager.PageTransformer {

            private static final float MIN_SCALE = 0.75f;

            @Override
            public void transformPage(View view, float position) {
                int pageWidth = view.getWidth();

                if (position < -1) { // [-Infinity,-1)
                    // This page is way off-screen to the left.
                    view.setAlpha(0);

                } else if (position <= 0) { // [-1,0]
                    // Use the default slide transition when moving to the left page
                    view.setAlpha(1);
                    view.setTranslationX(0);
                    view.setScaleX(1);
                    view.setScaleY(1);

                } else if (position <= 1) { // (0,1]
                    // Fade the page out.
                    view.setAlpha(1 - position);

                    // Counteract the default slide transition
                    view.setTranslationX(pageWidth * -position);

                    // Scale the page down (between MIN_SCALE and 1)
                    float scaleFactor = MIN_SCALE
                            + (1 - MIN_SCALE) * (1 - Math.abs(position));
                    view.setScaleX(scaleFactor);
                    view.setScaleY(scaleFactor);

                } else { // (1,+Infinity]
                    // This page is way off-screen to the right.
                    view.setAlpha(0);
                }
            }
        }

        public class DefaultPageTransformer implements ViewPager.PageTransformer {

            @Override
            public void transformPage(View view, float position) {
                if (position < -1) { // [-Infinity,-1)
                    // This page is way off-screen to the left.
                    view.setAlpha(0);
                } else if (position <= 1) { // [-1,0]
                    // Use the default slide transition when moving to the left page
                    view.setAlpha(1);
                    view.setTranslationX(0);
                    view.setScaleX(1);
                    view.setScaleY(1);
                } else { // (1,+Infinity]
                    // This page is way off-screen to the right.
                    view.setAlpha(0);
                }
            }
        }

        public class DepthSuperPageTransformer implements ViewPager.PageTransformer {
            private static final float MIN_SCALE = 0.75f;

            @Override
            public void transformPage(View view, float position) {
                int pageWidth = view.getWidth();
                if (position < -1) { // [-Infinity,-1)
                    // This page is way off-screen to the left.
                    view.setAlpha(0);
                } else if (position <= 1) { // [-1,0]
                    // Fade the page out.
                    view.setAlpha(1 - Math.abs(position));

                    // Counteract the default slide transition
                    view.setTranslationX(pageWidth * -Math.abs(position));

                    // Scale the page down (between MIN_SCALE and 1)
                    float scaleFactor = MIN_SCALE
                            + (1 - MIN_SCALE) * (1 - Math.abs(position));
                    view.setScaleX(scaleFactor);
                    view.setScaleY(scaleFactor);
                } else { // (1,+Infinity]
                    // This page is way off-screen to the right.
                    view.setAlpha(0);
                }
            }
        }
    }
}
