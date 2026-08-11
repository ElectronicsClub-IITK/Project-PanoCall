import 'dart:async';

import '../models/imu_data.dart';
import 'performance_service.dart';
import 'orientation_service.dart';
import 'native_sensor_service.dart';

/// Reads IMU data from the native Android sensor bridge (see
/// android_native/HighRateImuStreamHandler.kt) instead of sensors_plus.
///
/// sensors_plus tops out around 50-60Hz in practice because each of
/// accel/gyro/mag streams over its own EventChannel with per-message
/// overhead. The native bridge fuses all three into one packet and
/// pushes it over a single channel at a fixed, configurable rate
/// (default 200Hz), which is what actually gets you to 150-200Hz.
class ImuService {
  final ImuData imu = ImuData();

  final PerformanceService performance;
  final OrientationService orientation;
  final NativeSensorService _native = NativeSensorService();

  /// Target output rate from native side. Change to 150 if your device's
  /// hardware sensors can't sustain 200Hz cleanly (check logcat tag
  /// "ImuVR" for each sensor's real minDelay-based max rate).
  final int targetRateHz;

  ImuService({
    required this.performance,
    required this.orientation,
    
    this.targetRateHz = 200,
  });

  StreamSubscription<ImuData>? _nativeSub;

  Future<void> start() async {
    await _native.setOutputRate(targetRateHz);

    _nativeSub = _native.imuStream.listen(
      _onImuSample,
      onError: (Object e) {
        // ignore: avoid_print
        print("Native IMU stream error: $e");
      },
    );
  }

  void stop() {
    _nativeSub?.cancel();
    _nativeSub = null;
  }

  void _onImuSample(ImuData sample) {
    imu.ax = sample.ax;
    imu.ay = sample.ay;
    imu.az = sample.az;

    imu.gx = sample.gx;
    imu.gy = sample.gy;
    imu.gz = sample.gz;

    imu.mx = sample.mx;
    imu.my = sample.my;
    imu.mz = sample.mz;

    imu.timestamp = sample.timestamp;

    // Runs at full 150-200Hz — this is what your VR laptop-render pipeline
    // should read from / be fed over USB, since it needs the raw rate.
    performance.addSample();

    // Orientation integration also runs at full rate (more samples =
    // more accurate integration), but it no longer calls notifyListeners()
    // on every single call — see orientation_service.dart. The UI redraw
    // is throttled separately so 200Hz data doesn't mean 200Hz UI jank.
    orientation.updateGyro(
      gx: sample.gx,
      gy: sample.gy,
      gz: sample.gz,
    );
  }
}
