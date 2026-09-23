package com.example.icebergdss

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.View
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity

class MainActivity : ComponentActivity() {

    private lateinit var webView: WebView
    // Placeholder URL for the Netlify hosted site
    // You can replace this with your actual Netlify URL, e.g. "https://antarctic-dss.netlify.app"
    private val siteUrl = "https://sih26kryptonite.netlify.app/frontend/index.html" // Points to local emulator host by default if running backend locally, or replace with netlify URL.

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        webView = WebView(this)
        setContentView(webView)

        // Hide system UI for a fully immersive app experience
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_FULLSCREEN
                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY)

        setupWebView()
        
        // Load the web app
        webView.loadUrl(siteUrl)
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun setupWebView() {
        // Optimize WebSettings for maps and animations
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            loadWithOverviewMode = true
            useWideViewPort = true
            mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            // Disable zoom controls to make it feel like a native app instead of a browser
            setSupportZoom(false)
            builtInZoomControls = false
            displayZoomControls = false
        }

        // Over scroll mode to NEVER to prevent the page from bouncing (feels more native)
        webView.overScrollMode = View.OVER_SCROLL_NEVER

        // Bridge JS to Native
        webView.addJavascriptInterface(WebAppInterface(this, webView), "AndroidHaptics")

        // Inject JS to automatically hook existing HTML elements to native haptics
        webView.webViewClient = object : WebViewClient() {
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                injectHapticsBinding(view)
            }
        }

        // Set WebChromeClient for potential JS alerts and console logging
        webView.webChromeClient = WebChromeClient()
    }

    private fun injectHapticsBinding(view: WebView?) {
        val js = """
            javascript:(function() {
                // Check if AndroidHaptics is available
                if (typeof AndroidHaptics === 'undefined') return;

                // Helper to add click listeners safely
                function addHaptic(selector, type) {
                    document.querySelectorAll(selector).forEach(el => {
                        // Prevent adding multiple listeners if we re-inject
                        if (el.dataset.hapticBound) return;
                        el.dataset.hapticBound = 'true';
                        el.addEventListener('click', () => {
                            AndroidHaptics.vibrate(type);
                        });
                    });
                }

                // Bind standard buttons -> light click
                addHaptic('button, .pb-btn, .file-btn', 'light');

                // Bind major actions -> heavy click
                addHaptic('.rp-place-btn, .ct-refresh-btn, .scenario-btn', 'heavy');

                // Hook into Leaflet map clicks for haptics
                if (typeof L !== 'undefined') {
                    // Try to find the map instance from our DOM
                    setTimeout(() => {
                        // If mapCtrl exists globally (from app.js)
                        if (window._routePlanner && window._routePlanner.mapCtrl && window._routePlanner.mapCtrl.map) {
                            window._routePlanner.mapCtrl.map.on('click', () => {
                                AndroidHaptics.vibrate('light');
                            });
                        }
                    }, 1000);
                }

                // Bind scrubber -> tick (using input event)
                const scrubber = document.getElementById('timeScrubber');
                if (scrubber && !scrubber.dataset.hapticBound) {
                    scrubber.dataset.hapticBound = 'true';
                    scrubber.addEventListener('input', () => {
                        AndroidHaptics.vibrate('tick');
                    });
                }
            })();
        """.trimIndent()
        
        view?.evaluateJavascript(js, null)
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack()
        } else {
            @Suppress("DEPRECATION")
            super.onBackPressed()
        }
    }
}
