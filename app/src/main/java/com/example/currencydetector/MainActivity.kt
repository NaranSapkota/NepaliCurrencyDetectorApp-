package com.example.currencydetector

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.view.animation.AlphaAnimation
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.cardview.widget.CardView
import androidx.core.content.ContextCompat
import com.example.currencydetector.databinding.ActivityMainBinding
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var detector: ObjectDetector

    private var lastActionTime = 0L
    private val COOLDOWN_MS = 3000L

    // flashlight control
    private var cameraControl: CameraControl? = null
    private var cameraInfo: CameraInfo? = null
    private var isTorchOn = false

    private val audioMap: Map<String, Int> by lazy {
        mapOf(
            "five" to R.raw.five,
            "ten" to R.raw.ten,
            "twenty" to R.raw.twenty,
            "fifty" to R.raw.fifty,
            "hundred" to R.raw.hundred,
            "fivehundred" to R.raw.fivehundred,
            "thousand" to R.raw.thousand
        )
    }
    private val beepRes = R.raw.beep

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        loadModel()

        binding.btnFlash.setOnClickListener {
            toggleTorch()
        }

        binding.btnGallery.setOnClickListener {
            // open system gallery to Pictures/CurrencyCaptures (may vary by device)
            Toast.makeText(this, "Saved captures: Pictures/CurrencyCaptures (check your gallery)", Toast.LENGTH_LONG).show()
            // Optionally open gallery app
            val uri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            val intent = Intent(Intent.ACTION_VIEW, uri)
            startActivity(intent)
        }

        // request permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            // startCamera will be called after model loads
        } else permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun loadModel() {
        Thread {
            try {
                val options = ObjectDetectorOptions.builder()
                    .setMaxResults(1)
                    .setScoreThreshold(0.0f)
                    .build()

                detector = ObjectDetector.createFromFileAndOptions(
                    this, "NepaliCurrencyDetectorModel.tflite", options
                )
                runOnUiThread { Toast.makeText(this, "Model loaded", Toast.LENGTH_SHORT).show() }
                // start camera after model loaded and permission granted
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    == android.content.pm.PackageManager.PERMISSION_GRANTED
                ) startCamera()
            } catch (e: Exception) {
                Log.e("TFLITE", "Model load failed: $e")
                runOnUiThread {
                    Toast.makeText(this, "Model load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setTargetResolution(android.util.Size(640, 480))
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                analyzeImage(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            cameraControl = camera.cameraControl
            cameraInfo = camera.cameraInfo

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeImage(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        if (bitmap == null) {
            imageProxy.close()
            return
        }

        // Crop to scan box area to improve accuracy & performance
        val cropRect = getCropRectForScanBox(bitmap.width, bitmap.height)
        val cropped = Bitmap.createBitmap(bitmap, cropRect.left, cropRect.top, cropRect.width(), cropRect.height())

        val tensor = TensorImage.fromBitmap(cropped)

        try {
            val results = detector.detect(tensor)
            if (results.isNotEmpty()) {
                val cat = results[0].categories[0]
                val label = cat.label.lowercase(Locale.ROOT)
                val score = cat.score

                runOnUiThread {
                    binding.tvLabel.text = label.uppercase(Locale.ROOT)
                    binding.tvConfidence.text = "Confidence: ${(score * 100).toInt()}%"
                    val card = binding.predCard as CardView
                    if (score >= 0.80f) {
                        card.setCardBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_green_light))
                    } else {
                        card.setCardBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_red_light))
                    }
                    val anim = AlphaAnimation(0.3f, 1f)
                    anim.duration = 200
                    binding.predCard.startAnimation(anim)
                }

                val now = System.currentTimeMillis()
                if (score >= 0.80f && now - lastActionTime > COOLDOWN_MS) {
                    lastActionTime = now
                    runOnUiThread { showLoading(true) }

                    // beep + note audio
                    playRaw(beepRes)
                    audioMap[label]?.let { res ->
                        Thread { Thread.sleep(300); playRaw(res) }.start()
                    }

                    // save capture (full size)
                    saveBitmapToGallery(this, bitmap, label)

                    runOnUiThread {
                        binding.previewView.postDelayed({ showLoading(false) }, 1000)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("INFER", "Error: $e")
        } finally {
            imageProxy.close()
        }
    }

    private fun getCropRectForScanBox(w: Int, h: Int): android.graphics.Rect {
        val boxW = (260 * resources.displayMetrics.density).toInt()
        val boxH = (160 * resources.displayMetrics.density).toInt()
        val left = (w - boxW) / 2
        val top = (h - boxH) / 2
        val right = (left + boxW).coerceAtMost(w)
        val bottom = (top + boxH).coerceAtMost(h)
        return android.graphics.Rect(left.coerceAtLeast(0), top.coerceAtLeast(0), right, bottom)
    }

    private fun playRaw(resId: Int) {
        try {
            val mp = android.media.MediaPlayer.create(this, resId)
            mp.setOnCompletionListener { it.release() }
            mp.start()
        } catch (e: Exception) {
            Log.e("AUDIO", "Play error: $e")
        }
    }

    private fun showLoading(show: Boolean) {
        if (show) {
            binding.loadingOverlay.visibility = View.VISIBLE
            binding.loadingOverlay.animate().alpha(1f).setDuration(180).start()
        } else {
            binding.loadingOverlay.animate().alpha(0f).setDuration(180).withEndAction {
                binding.loadingOverlay.visibility = View.GONE
            }.start()
        }
    }

    // Save full bitmap to Pictures/CurrencyCaptures
    private fun saveBitmapToGallery(context: Context, bmp: Bitmap, label: String) {
        val filename = "capture_${label}_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg"
        var out: OutputStream? = null
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/CurrencyCaptures")
                }
                val uri: Uri? = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                uri?.let { out = contentResolver.openOutputStream(it) }
            } else {
                val imagesDir = getExternalFilesDir("Pictures/CurrencyCaptures")
                val file = java.io.File(imagesDir, filename)
                out = java.io.FileOutputStream(file)
            }
            out?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }
            runOnUiThread { Toast.makeText(this, "Saved: $filename", Toast.LENGTH_SHORT).show() }
        } catch (e: Exception) {
            Log.e("SAVE", "Save failed: $e")
        } finally {
            try { out?.close() } catch (_: Exception) {}
        }
    }

    private fun toggleTorch() {
        cameraControl?.enableTorch(!isTorchOn)
        isTorchOn = !isTorchOn
        val toast = if (isTorchOn) "Flash ON" else "Flash OFF"
        Toast.makeText(this, toast, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
