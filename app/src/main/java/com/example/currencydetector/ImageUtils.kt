package com.example.currencydetector

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

fun ImageProxy.toBitmap(): Bitmap? {
    return try {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        // NV21 ordering: Y, V, U
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 90, out)
        val imageBytes = out.toByteArray()
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    } catch (e: Exception) {
        null
    }
}
