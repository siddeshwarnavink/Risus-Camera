package com.sidapps.risuscamera

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutionException
import kotlin.math.max
import kotlin.math.min


class SmileDetector(assetManager: AssetManager) {
    private lateinit var interpreter: Interpreter

    init {
        loadModel(assetManager)
    }

    private fun loadModel(assetManager: AssetManager) {
        val tfliteOptions = Interpreter.Options()
        val modelFileDescriptor = assetManager.openFd("smile.tflite")
        val inputStream = FileInputStream(modelFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = modelFileDescriptor.startOffset
        val declaredLength = modelFileDescriptor.declaredLength
        val modelByteBuffer =
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(modelByteBuffer, tfliteOptions)
    }

    interface DetectionCallback {
        fun onFaceDetected(bitmap: Bitmap, isSmiling: Boolean)
    }

    fun detectSmile(bitmap: Bitmap, callback: DetectionCallback) {
        val faceBitmap = detectAndCropFace(bitmap) ?: return

        val inputBuffer = preprocessImage(faceBitmap)
        val output = Array(1) { FloatArray(1) }
        interpreter.run(inputBuffer, output)
        val smileProbability = output[0][0]
        Log.d("SmileDetector", "Smile Probability: $smileProbability")

        callback.onFaceDetected(faceBitmap, smileProbability >= 0.8f)
    }

    private fun detectAndCropFace(bitmap: Bitmap): Bitmap? {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .build()

        val detector = FaceDetection.getClient(options)
        val image = InputImage.fromBitmap(bitmap, 0)
        try {
            val faces = Tasks.await(detector.process(image))
            if (faces.isNotEmpty()) {
                val face = faces[0]
                val bounds = face.boundingBox

                val left = max(0, bounds.left)
                val top = max(0, bounds.top)
                val right = min(bitmap.width, bounds.right)
                val bottom = min(bitmap.height, bounds.bottom)
                val width = right - left
                val height = bottom - top

                if (width > 0 && height > 0) {
                    return Bitmap.createBitmap(bitmap, left, top, width, height)
                }
            }
        } catch (e: ExecutionException) {
            Log.e("SmileDetector", "Failed to detect faces", e)
        } catch (e: InterruptedException) {
            Log.e("SmileDetector", "Face detection interrupted", e)
        }

        return null
    }


    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true)

        // Convert the bitmap to grayscale
        val grayBitmap = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f)
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(resizedBitmap, 0f, 0f, paint)

        val inputBuffer = ByteBuffer.allocateDirect(4 * 32 * 32)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 32) {
            for (x in 0 until 32) {
                val pixelValue = grayBitmap.getPixel(x, y)
                val r = (pixelValue shr 16) and 0xFF
                val normalizedPixelValue =
                    r / 255.0f
                inputBuffer.putFloat(normalizedPixelValue)
            }
        }

        return inputBuffer
    }

}
