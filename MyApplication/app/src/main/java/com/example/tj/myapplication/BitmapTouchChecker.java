package com.example.tj.myapplication;

import android.graphics.Bitmap;
import android.util.Log;
import com.example.tj.myapplication.PanButton.TouchChecker;

/**
 * Created by tj on 15-2-10.
 */
public class BitmapTouchChecker implements TouchChecker
{
    private Bitmap bitmap;

    public BitmapTouchChecker(Bitmap bitmap)
    {
        this.bitmap = bitmap;
    }

    @Override
    public boolean isInTouchArea(int x, int y, int width, int height)
    {
        if (bitmap != null)
        {
            int pixel = bitmap.getPixel(x, y);

            if (((pixel >> 24) & 0xff) > 0)
            {
                Log.d("BitmapTouchChecker", "isInTouchArea return true");

                return true;
            }
        }

        Log.d("BitmapTouchChecker", "isInTouchArea return false");

        return false;
    }

}
