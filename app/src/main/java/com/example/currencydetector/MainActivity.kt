package com.example.currencydetector

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.currencydetector.databinding.ActivityMainBinding
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var detector: ObjectDetector? = null

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Camera permission denied", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        loadModel()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            // startCamera() will be called after the model loads successfully
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun getModelFile(context: Context, modelName: String): File {
        val modelFile = File(context.filesDir, modelName)
        if (!modelFile.exists()) {
            try {
                context.assets.open(modelName).use { inputStream ->
                    FileOutputStream(modelFile).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
            } catch (e: IOException) {
                throw RuntimeException("Error copying model to internal storage", e)
            }
        }
        return modelFile
    }

    private fun loadModel() {
        Thread {
            try {
                val modelFile = getModelFile(this, "NepaliCurrencyDetectorModel.tflite")

                val options = ObjectDetectorOptions.builder()
                    .setMaxResults(1)
                    .setScoreThreshold(0.6f) // Increased threshold for better accuracy
                    .build()

                detector = ObjectDetector.createFromFileAndOptions(modelFile, options)

                runOnUiThread {
                    Toast.makeText(this, "Model Loaded!", Toast.LENGTH_SHORT).show()
                    // Now that the model is loaded, we can start the camera if we have permission
                    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                        == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                        startCamera()
                    }
                }

            } catch (e: Exception) {
                Log.e("MODEL", "Load fail: $e")
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

            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            analyzer.setAnalyzer(cameraExecutor) { imageProxy ->
                analyze(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, analyzer)
            } catch (e: Exception) {
                Log.e("CAMERA", "Bind fail: $e")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(imageProxy: ImageProxy) {
        if (detector == null) {
            imageProxy.close()
            return
        }

        // 1. Convert YUV to Bitmap
        val bitmap = imageProxy.toBitmap() ?: run {
            imageProxy.close()
            return
        }

        // 2. Rotate the image for the model
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val imageProcessor = ImageProcessor.Builder().add(Rot90Op(-rotationDegrees / 90)).build()
        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(bitmap))

        try {
            // 3. Run detection
            val results = detector!!.detect(tensorImage)

            if (results.isNotEmpty() && results[0].categories.isNotEmpty()) {
                val obj = results[0]
                val category = obj.categories[0]
                val label = category.label.lowercase(Locale.ROOT)
                val confidence = (category.score * 100).toInt()

                runOnUiThread {
                    binding.tvLabel.text = label.uppercase()
                    binding.tvConfidence.text = "Confidence: $confidence%"
                }
            } else {
                runOnUiThread {
                    binding.tvLabel.text = "SCANNING..."
                    binding.tvConfidence.text = "Confidence: --%"
                }
            }

        } catch (e: Exception) {
            Log.e("DETECT", "Error: $e")
        } finally {
            imageProxy.close()
        }
    }

    // *** THE CORRECT IMPLEMENTATION FOR toBitmap() ***
    private fun ImageProxy.toBitmap(): Bitmap? {
        if (format != ImageFormat.YUV_420_888) {
            Log.e("Bitmap", "Unsupported image format: $format")
            return null
        }

        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // U and V are swapped in NV21 format compared to I420 format
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val jpegBytes = out.toByteArray()

        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        detector?.close()
    }
}