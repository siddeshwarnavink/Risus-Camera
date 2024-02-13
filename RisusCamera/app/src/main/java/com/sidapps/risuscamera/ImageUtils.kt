package com.sidapps.risuscamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicYuvToRGB
import android.renderscript.Type
import android.util.Log

object ImageUtils {
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

}
