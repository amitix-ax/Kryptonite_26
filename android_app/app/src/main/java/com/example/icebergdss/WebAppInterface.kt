package com.example.icebergdss

import android.content.Context
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.view.HapticFeedbackConstants
import android.view.View
import android.webkit.JavascriptInterface

class WebAppInterface(private val mContext: Context, private val webView: View) {

    private val vibrator: Vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        val vibratorManager = mContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        vibratorManager.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        mContext.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
    }

    /** Show a toast from the web page */
    @JavascriptInterface
    fun showToast(toast: String) {
        android.widget.Toast.makeText(mContext, toast, android.widget.Toast.LENGTH_SHORT).show()
    }

    @JavascriptInterface
    fun vibrate(type: String) {
        webView.post {
            when (type.lowercase()) {
                "light" -> {
                    // Light click for standard buttons
                    webView.performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                }
                "heavy" -> {
                    // Heavy click for map waypoints and scenarios
                    webView.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                }
                "tick" -> {
                    // Fast tick for scrubber/timeline
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        vibrator.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_TICK))
                    } else {
                        webView.performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
                    }
                }
                "error" -> {
                    // Double buzz for critical hazards
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        vibrator.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_DOUBLE_CLICK))
                    } else {
                        webView.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                    }
                }
                else -> {
                    // Default
                    webView.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY)
                }
            }
        }
    }
}
