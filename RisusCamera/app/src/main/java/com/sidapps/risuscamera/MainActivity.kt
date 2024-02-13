package com.sidapps.risuscamera

import kotlin.math.max
import kotlin.math.min

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
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.sidapps.risuscamera.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(angle) }
    return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
}

fun Image.yuvToRgb(context: Context): Bitmap? {
    if (format != ImageFormat.YUV_420_888 || planes.size != 3) {
        Log.e("ImageConversion", "Unsupported image format")
        return null
    }

    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val rs = RenderScript.create(context)
    val scriptYuvToRgb = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs))

    val yuvType = Type.Builder(rs, Element.U8(rs)).setX(nv21.size)
    val inAllocation = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT)
    val outBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    val outAllocation = Allocation.createFromBitmap(rs, outBitmap)

    inAllocation.copyFrom(nv21)

    scriptYuvToRgb.setInput(inAllocation)
    scriptYuvToRgb.forEach(outAllocation)

    outAllocation.copyTo(outBitmap)

    inAllocation.destroy()
    outAllocation.destroy()
    scriptYuvToRgb.destroy()
    rs.destroy()

    return outBitmap
}

@ExperimentalGetImage
class MainActivity : AppCompatActivity() {
    private lateinit var viewBinding: ActivityMainBinding
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var interpreter: Interpreter

    private var lastPhotoTime = 0L

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
        val context = this@MainActivity
        val bitmap = imageProxy.image?.yuvToRgb(context)

        bitmap?.let {
            val rotatedBitmap = rotateBitmap(bitmap, -90f)

            smileDetector.detectSmile(rotatedBitmap, object : SmileDetector.DetectionCallback {
                override fun onFaceDetected(bitmap: Bitmap, isSmiling: Boolean) {
                    runOnUiThread {
                        if (isSmiling) {
                            //viewBinding.facePreview.setImageBitmap(rotatedBitmap)
                            takePhoto()
                        }
                    }
                }
            })
        }


        imageProxy.close()
    }


    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun takePhoto() {
        val currentTime = System.currentTimeMillis()
        if (currentTime - lastPhotoTime < 3000) return
        lastPhotoTime = currentTime

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
