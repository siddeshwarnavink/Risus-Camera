package com.sidapps.risuscamera

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.media.Image
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.renderscript.Allocation
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.renderscript.Element
import android.renderscript.Type
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.sidapps.risuscamera.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

object BitmapUtils {
    fun byteArrayToBitmap(bytes: ByteArray, width: Int, height: Int): Bitmap? {
        val bytesPerPixel = 4
        val byteArraySize = bytes.size
        val expectedSize = width * height * bytesPerPixel

        Log.d("BitmapUtils", "Byte array size: $byteArraySize")
        Log.d("BitmapUtils", "Expected size: $expectedSize")
        Log.d("BitmapUtils", "Width: $width, Height: $height")

        if (byteArraySize < expectedSize) {
            Log.e("BitmapUtils", "Insufficient data in byte array to create bitmap")
            return null
        }

        return try {
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
                copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(bytes))
            }
        } catch (e: Exception) {
            Log.e("BitmapUtils", "Failed to create bitmap: ${e.message}")
            null
        }
    }

    fun rotateBitmap(bitmap: Bitmap?, degrees: Float): Bitmap? {
        if (bitmap == null) {
            return null
        }

        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }
}

fun Image.yuvToRgb(context: Context): Bitmap? {
    // Ensure the image format is YUV_420_888
    if (format != ImageFormat.YUV_420_888 || planes.size != 3) {
        Log.e("ImageConversion", "Unsupported image format")
        return null
    }

    // Get the pixel data from the Image
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    // Get the image dimensions
    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    // Create a byte array for the YUV data
    val nv21 = ByteArray(ySize + uSize + vSize)

    // Copy the YUV data into the array
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    // Prepare the RenderScript
    val rs = RenderScript.create(context)
    val scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    // Create allocations for input and output
    val yuvType = Type.Builder(rs, Element.U8(rs)).setX(nv21.size)
    val inAllocation = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)
    val outBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val outAllocation = Allocation.createFromBitmap(rs, outBitmap)

    // Set the YUV data to the input allocation
    inAllocation.copyFrom(nv21)

    // Convert YUV to RGB
    scriptYuvToRgb.setInput(inAllocation)
    scriptYuvToRgb.forEach(outAllocation)

    // Copy the RGB data to the bitmap
    outAllocation.copyTo(outBitmap)

    // Clean up
    inAllocation.destroy()
    outAllocation.destroy()
    scriptYuvToRgb.destroy()
    rs.destroy()

    return outBitmap
}

fun Image.toBitmap(rotationDegrees: Int): Bitmap? {
    val buffer = planes[0].buffer
    val bytes = ByteArray(buffer.remaining())
    buffer.get(bytes)

    val width = planes[0].rowStride / planes[0].pixelStride
    val height = height

    Log.d("MainActivity", "Image width: $width, height: $height")

    val byteArraySize = bytes.size
    Log.d("MainActivity", "Byte array size: $byteArraySize")

    return if (rotationDegrees == 0 || rotationDegrees == 180) {
        BitmapUtils.byteArrayToBitmap(bytes, width, height)
    } else {
        // Rotate the bitmap
        val rotatedBitmap = BitmapUtils.byteArrayToBitmap(bytes, width, height)
        BitmapUtils.rotateBitmap(rotatedBitmap, rotationDegrees.toFloat())
    }
}

@ExperimentalGetImage
class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var interpreter: Interpreter

    private val smileDetector: SmileDetector by lazy {
        SmileDetector(assets)
    }

    private val activityResultLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        )
        { permissions ->
            // Handle Permission granted/rejected
            var permissionGranted = true
            permissions.entries.forEach {
                if (it.key in REQUIRED_PERMISSIONS && it.value == false)
                    permissionGranted = false
            }
            if (!permissionGranted) {
                Toast.makeText(
                    baseContext,
                    "Permission request denied",
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                startCamera()
            }
        }

    override fun onStop() {
        super.onStop()
        interpreter.close()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)

        // Check if the device has a camera
        if (!packageManager.hasSystemFeature(PackageManager.FEATURE_CAMERA_ANY)) {
            Toast.makeText(this, "No camera found", Toast.LENGTH_SHORT).show()
            return
        }

        // Request camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            requestPermissions()
        }

        viewBinding.imageCaptureButton.setOnClickListener { takePhoto() }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun requestPermissions() {
        activityResultLauncher.launch(REQUIRED_PERMISSIONS)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            val cameraProvider = cameraProviderFuture.get()

            val preview: Preview = Preview.Builder()
                .build()

            imageCapture = ImageCapture.Builder()
                .build()

            val cameraSelector: CameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()


            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this as LifecycleOwner,
                    cameraSelector,
                    preview,
                    imageCapture
                )

                val imageAnalysis = ImageAnalysis.Builder()
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor, smileAnalyzer)
                    }
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview)
            } catch (exc: Exception) {
                Log.e("TAG", "Use case binding failed", exc)
            }

            preview.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private val smileAnalyzer = ImageAnalysis.Analyzer { imageProxy ->
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val context = this@MainActivity
        val bitmap = imageProxy.image?.yuvToRgb(context)

        bitmap?.let {
            val result = smileDetector.detectSmile(it)
            if (result) {
                takePhoto()
            }
        }

        imageProxy.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        val name = SimpleDateFormat(FILENAME_FORMAT, Locale.US)
            .format(System.currentTimeMillis())

        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, name)
            put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT > Build.VERSION_CODES.P) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Reisus-Image")
            }
        }

        val outputOptions = ImageCapture.OutputFileOptions
            .Builder(
                contentResolver,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )
            .build()

        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onError(exc: ImageCaptureException) {
                    Log.e("TAG", "Photo capture failed: ${exc.message}", exc)
                }

                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    val msg = "Photo capture!"
                    Toast.makeText(baseContext, msg, Toast.LENGTH_SHORT).show()
                    Log.d("TAG", msg + name)
                }
            }
        )
    }

    companion object {
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS =
            mutableListOf(
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO
            ).apply {
                if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
                    add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                }
            }.toTypedArray()
    }
}

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

    fun detectSmile(bitmap: Bitmap): Boolean {
        val inputBuffer = preprocessImage(bitmap)
        val output = Array(1) { FloatArray(1) }
        interpreter.run(inputBuffer, output)
        val smileProbability = output[0][0]
        Log.d("SmileDetector", "Smile Probability: $smileProbability")
        return smileProbability >= 0.7f
    }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize the bitmap to 32x32 pixels
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 32, 32, true)

        // Convert the bitmap to grayscale
        val grayBitmap = Bitmap.createBitmap(32, 32, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(grayBitmap)
        val paint = Paint()
        val colorMatrix = ColorMatrix()
        colorMatrix.setSaturation(0f) // Set saturation to 0 to create grayscale
        val filter = ColorMatrixColorFilter(colorMatrix)
        paint.colorFilter = filter
        canvas.drawBitmap(resizedBitmap, 0f, 0f, paint)

        // Allocate the ByteBuffer for TensorFlow Lite model
        val inputBuffer = ByteBuffer.allocateDirect(4 * 32 * 32)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Populate the ByteBuffer with pixel values
        for (y in 0 until 32) {
            for (x in 0 until 32) {
                val pixelValue = grayBitmap.getPixel(x, y)
                val r = (pixelValue shr 16) and 0xFF
                val normalizedPixelValue =
                    r / 255.0f // Assuming the model expects pixel values between 0 and 1
                inputBuffer.putFloat(normalizedPixelValue)
            }
        }

        return inputBuffer
    }

}
