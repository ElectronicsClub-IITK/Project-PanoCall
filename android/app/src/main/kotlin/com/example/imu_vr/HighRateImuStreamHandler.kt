package com.example.imu_vr   // <-- CHANGE to match your applicationId / existing package

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.view.Surface
import android.view.WindowManager
import io.flutter.plugin.common.EventChannel

/**
 * Reads accelerometer / gyroscope / magnetometer directly from Android's
 * SensorManager (bypassing the sensors_plus per-sensor EventChannel model),
 * and pumps ONE fused, timestamped packet to Dart at a fixed output rate.
 *
 * Also remaps the raw sensor axes to match the CURRENT screen rotation.
 * Android's sensor axes are fixed to the physical device body, not to
 * whatever orientation the screen is locked to — without this remap,
 * turning left/right in landscape reads as pitch instead of yaw (and
 * vice versa). This uses the same remap pattern Android's own docs use
 * for compass/tilt apps.
 *
 * Why this fixes the 50Hz ceiling:
 *  - sensors_plus's `gameInterval` IS 50Hz by definition (20ms). Even its
 *    faster options are still bottlenecked by 3 separate EventChannels,
 *    each with per-message serialization overhead.
 *  - Here we register listeners at true hardware speed (samplingPeriodUs),
 *    cache the latest value per axis-group off the UI thread, and flush a
 *    single merged packet over ONE channel at a controlled rate. That's
 *    1 channel crossing per tick instead of ~3, and it's rate-limited by
 *    us, not by however fast the OS decides to fire raw sensor callbacks.
 */
class HighRateImuStreamHandler(
    private val sensorManager: SensorManager,
    private val context: Context
) : EventChannel.StreamHandler, SensorEventListener {

    // ------------------------------------------------------------
    // Crash/error logging that doesn't need adb, developer mode, or a
    // file manager. Writes to this app's private internal storage —
    // Android 11+ blocks external file managers from browsing
    // Android/data/ anyway, so internal storage + an in-app viewer
    // (see MainActivity's "readLog" method + DebugLogScreen in Dart)
    // is the reliable way to see this now.
    // ------------------------------------------------------------
    private fun logToFile(message: String) {
        try {
            val file = java.io.File(context.filesDir, "imu_vr_log.txt")
            file.appendText("${java.util.Date()} — $message\n")
        } catch (e: Exception) {
            // If even logging fails, there's nothing more we can safely do here.
        }
        android.util.Log.e("ImuVR", message)
    }

    private var eventSink: EventChannel.EventSink? = null

    private val accel = FloatArray(3)
    private val gyro = FloatArray(3)
    private val mag = FloatArray(3)

    private var hasAccel = false
    private var hasGyro = false
    private var hasMag = false

    // Sensor callbacks run here, off the UI thread, so they never get
    // starved/delayed by Flutter frame rendering.
    private val sensorThread = HandlerThread("ImuSensorThread").apply { start() }
    private val sensorHandler = Handler(sensorThread.looper)

    // Fixed-rate output pump, also off the UI thread.
    private val outputThread = HandlerThread("ImuOutputThread").apply { start() }
    private val outputHandler = Handler(outputThread.looper)
    private val mainHandler = Handler(Looper.getMainLooper())

    // ---- TUNE THIS: output rate sent to Dart ----
    // 5ms  -> 200 Hz
    // 6.7ms-> 150 Hz
    private var outputPeriodMs = 5L

    // Requested hardware sampling period (we ask for slightly faster than
    // output rate so we always have a fresh sample when the pump fires).
    private var requestedSamplingPeriodUs = 4000 // 250Hz request -> device gives its max if lower

    private val pumpRunnable = object : Runnable {
        override fun run() {
            try {
                emitSample()
            } catch (e: Exception) {
                logToFile("emitSample failed: ${e.message}\n${e.stackTraceToString()}")
            }
            outputHandler.postDelayed(this, outputPeriodMs)
        }
    }

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        eventSink = events
        try {
            registerSensors()
            outputHandler.postDelayed(pumpRunnable, outputPeriodMs)
        } catch (e: Exception) {
            logToFile("onListen failed: ${e.message}\n${e.stackTraceToString()}")
        }
    }

    override fun onCancel(arguments: Any?) {
        outputHandler.removeCallbacks(pumpRunnable)
        sensorManager.unregisterListener(this)
        eventSink = null
    }

    private fun registerSensors() {
        val accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

        // Definitive check — logs exactly whether the manifest permission
        // actually took effect, instead of inferring it from a caught
        // exception further down.
        val permissionGranted = try {
            context.packageManager.checkPermission(
                "android.permission.HIGH_SAMPLING_RATE_SENSORS",
                context.packageName
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        } catch (e: Exception) {
            false
        }
        logToFile("HIGH_SAMPLING_RATE_SENSORS permission granted: $permissionGranted")
        logToFile("Requested sampling period: ${requestedSamplingPeriodUs}us " +
                "(${1_000_000.0 / requestedSamplingPeriodUs}Hz)")

        // minDelay (microseconds) tells you the fastest this device's sensor
        // hardware can actually go. Log it so you can see your real ceiling.
        //
        // NOTE: registering above 200Hz without the HIGH_SAMPLING_RATE_SENSORS
        // permission (Android 12+) throws a SecurityException here — wrapped
        // in try/catch so that shows up in the log file instead of closing
        // the whole app.
        try {
            accelSensor?.let {
                android.util.Log.i("ImuVR", "Accel minDelay=${it.minDelay}us (max ${1_000_000.0 / it.minDelay}Hz)")
                sensorManager.registerListener(this, it, requestedSamplingPeriodUs, 0, sensorHandler)
            }
            gyroSensor?.let {
                android.util.Log.i("ImuVR", "Gyro minDelay=${it.minDelay}us (max ${1_000_000.0 / it.minDelay}Hz)")
                sensorManager.registerListener(this, it, requestedSamplingPeriodUs, 0, sensorHandler)
            }
            magSensor?.let {
                android.util.Log.i("ImuVR", "Mag minDelay=${it.minDelay}us (max ${1_000_000.0 / it.minDelay}Hz)")
                sensorManager.registerListener(this, it, requestedSamplingPeriodUs, 0, sensorHandler)
            }
        } catch (e: SecurityException) {
            logToFile("SecurityException registering sensors at ${requestedSamplingPeriodUs}us — " +
                    "likely missing android.permission.HIGH_SAMPLING_RATE_SENSORS in AndroidManifest.xml. " +
                    "Falling back to a slower, always-allowed rate. Error: ${e.message}")
            // Fall back to a rate that's always allowed without the permission, so
            // at least something works while you add the permission properly.
            val safePeriodUs = 20000 // 50Hz, safe without the special permission
            try {
                accelSensor?.let { sensorManager.registerListener(this, it, safePeriodUs, 0, sensorHandler) }
                gyroSensor?.let { sensorManager.registerListener(this, it, safePeriodUs, 0, sensorHandler) }
                magSensor?.let { sensorManager.registerListener(this, it, safePeriodUs, 0, sensorHandler) }
            } catch (e2: Exception) {
                logToFile("Fallback sensor registration also failed: ${e2.message}")
            }
        } catch (e: Exception) {
            logToFile("Unexpected error registering sensors: ${e.message}\n${e.stackTraceToString()}")
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                accel[0] = event.values[0]; accel[1] = event.values[1]; accel[2] = event.values[2]
                hasAccel = true
            }
            Sensor.TYPE_GYROSCOPE -> {
                gyro[0] = event.values[0]; gyro[1] = event.values[1]; gyro[2] = event.values[2]
                hasGyro = true
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                mag[0] = event.values[0]; mag[1] = event.values[1]; mag[2] = event.values[2]
                hasMag = true
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --------------------------------------------------------------
    // Screen-rotation-aware axis remap
    // --------------------------------------------------------------

    private fun getDisplayRotation(): Int {
        return try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                context.display?.rotation ?: Surface.ROTATION_0
            } else {
                @Suppress("DEPRECATION")
                (context.getSystemService(Context.WINDOW_SERVICE) as WindowManager)
                    .defaultDisplay.rotation
            }
        } catch (e: Exception) {
            Surface.ROTATION_0
        }
    }

    private val remappedAccel = FloatArray(3)
    private val remappedGyro = FloatArray(3)

    /** Rotates a device-frame vector into the current screen's frame.
     * Same math Android's remapCoordinateSystem(AXIS_Y, AXIS_MINUS_X) etc.
     * would do — implemented manually because SensorManager's own version
     * throws an ArrayIndexOutOfBoundsException on this device/ROM. This
     * has no OS library dependency, so it can't hit that bug. */
    private fun remapForScreenRotation(input: FloatArray, output: FloatArray, rotation: Int) {
        val x = input[0]
        val y = input[1]
        val z = input[2]
        when (rotation) {
            Surface.ROTATION_90 -> {
                output[0] = y
                output[1] = -x
                output[2] = z
            }
            Surface.ROTATION_180 -> {
                output[0] = -x
                output[1] = -y
                output[2] = z
            }
            Surface.ROTATION_270 -> {
                output[0] = -y
                output[1] = x
                output[2] = z
            }
            else -> { // ROTATION_0
                output[0] = x
                output[1] = y
                output[2] = z
            }
        }
    }

    private fun emitSample() {
        if (eventSink == null) return
        if (!hasAccel && !hasGyro && !hasMag) return

        val rotation = getDisplayRotation()
        remapForScreenRotation(accel, remappedAccel, rotation)
        remapForScreenRotation(gyro, remappedGyro, rotation)

        val packet = HashMap<String, Any>()
        packet["timestamp"] = System.nanoTime() / 1000L // microseconds, matches ImuData.timestamp
        packet["ax"] = remappedAccel[0].toDouble(); packet["ay"] = remappedAccel[1].toDouble(); packet["az"] = remappedAccel[2].toDouble()
        packet["gx"] = remappedGyro[0].toDouble(); packet["gy"] = remappedGyro[1].toDouble(); packet["gz"] = remappedGyro[2].toDouble()
        packet["mx"] = mag[0].toDouble(); packet["my"] = mag[1].toDouble(); packet["mz"] = mag[2].toDouble()

        // EventSink.success MUST be called on the platform (UI) thread.
        mainHandler.post {
            eventSink?.success(packet)
        }
    }

    fun readLogFile(): String {
        return try {
            val file = java.io.File(context.filesDir, "imu_vr_log.txt")
            if (file.exists()) file.readText() else "(no log entries yet)"
        } catch (e: Exception) {
            "Failed to read log: ${e.message}"
        }
    }

    fun clearLogFile() {
        try {
            java.io.File(context.filesDir, "imu_vr_log.txt").delete()
        } catch (e: Exception) { }
    }

    /** Change output rate at runtime, e.g. from a MethodChannel call. */
    fun setOutputRateHz(hz: Int) {
        outputPeriodMs = (1000.0 / hz).toLong().coerceAtLeast(1)
    }
}
