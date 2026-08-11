package com.example.imu_vr   // <-- CHANGE to match your existing MainActivity's package

import android.content.Context
import android.hardware.SensorManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private val methodChannelName = "imu_vr/native"
    private val eventChannelName = "imu_vr/native_stream"

    private lateinit var streamHandler: HighRateImuStreamHandler

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        streamHandler = HighRateImuStreamHandler(sensorManager, this)

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, eventChannelName)
            .setStreamHandler(streamHandler)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannelName)
            .setMethodCallHandler { call, result ->
                try {
                    when (call.method) {
                        "testConnection" -> result.success("Native Android sensor bridge connected ✅")
                        "setOutputRateHz" -> {
                            val hz = call.argument<Int>("hz") ?: 200
                            streamHandler.setOutputRateHz(hz)
                            result.success(true)
                        }
                        "readLog" -> result.success(streamHandler.readLogFile())
                        "clearLog" -> {
                            streamHandler.clearLogFile()
                            result.success(true)
                        }
                        else -> result.notImplemented()
                    }
                } catch (e: Exception) {
                    try {
                        java.io.File(filesDir, "imu_vr_log.txt")
                            .appendText("${java.util.Date()} — MethodChannel error: ${e.message}\n${e.stackTraceToString()}\n")
                    } catch (_: Exception) { }
                    result.error("NATIVE_ERROR", e.message, null)
                }
            }
    }
}
