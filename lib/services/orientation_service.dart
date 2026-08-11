import 'dart:async';
import 'package:flutter/foundation.dart';

class OrientationService extends ChangeNotifier {
  // ------------------------------------------------------------
  // Orientation angles (degrees)
  // ------------------------------------------------------------

  double yaw = 0.0;   // Y-axis rotation
  double pitch = 0.0; // X-axis rotation
  double roll = 0.0;  // Z-axis rotation

  // ------------------------------------------------------------
  // Timing
  // ------------------------------------------------------------

  int _lastTimestampUs = 0;

  // ------------------------------------------------------------
  // Sensitivity control (VERY important for VR tuning)
  // ------------------------------------------------------------

  double sensitivity = 1.0;

  // ------------------------------------------------------------
  // UI refresh throttling
  //
  // updateGyro() can now be called at 150-200Hz (from the native sensor
  // bridge). Integrating the angle at that rate is good (more accurate),
  // but calling notifyListeners() that often forces Flutter to rebuild
  // the whole widget tree 150-200 times/sec, which is what causes jank,
  // not the sensor rate itself. So: integrate every call, but only
  // notify listeners on a fixed timer at a sane UI rate.
  // ------------------------------------------------------------

  static const int uiRefreshHz = 30;
  Timer? _uiTimer;

  OrientationService() {
    _uiTimer = Timer.periodic(
      Duration(milliseconds: (1000 / uiRefreshHz).round()),
      (_) => notifyListeners(),
    );
  }

  // ------------------------------------------------------------
  // Reset orientation
  // ------------------------------------------------------------

  void reset() {
    yaw = 0;
    pitch = 0;
    roll = 0;
    _lastTimestampUs = 0;
    notifyListeners(); // instant feedback — don't wait for the 30Hz UI timer
  }

  // ------------------------------------------------------------
  // Gyroscope update (CORE FUNCTION) — call this as fast as your
  // sensor feed provides samples (150-200Hz). No notifyListeners()
  // here on purpose; see uiRefreshHz timer above.
  // ------------------------------------------------------------

  // Gyroscope values arrive in radians/second (Android's native unit),
  // but yaw/pitch/roll here are in degrees — this conversion factor is
  // required, or a full 360° physical turn only moves the camera ~6°
  // (2π "degrees" instead of 360).
  static const double _radToDeg = 180.0 / 3.14159265358979323846;

  void updateGyro({
    required double gx,
    required double gy,
    required double gz,
  }) {
    final now = DateTime.now().microsecondsSinceEpoch;

    if (_lastTimestampUs != 0) {
      final dt = (now - _lastTimestampUs) / 1e6; // seconds

      // Integrate angular velocity → angle, converting rad/s to deg/s
      yaw += -gy * dt * sensitivity * _radToDeg;
      pitch += -gx * dt * sensitivity * _radToDeg;
      roll += -gz * dt * sensitivity * _radToDeg;

      // Keep angles in range (-180 to 180)
      yaw = _normalize(yaw);
      pitch = _normalize(pitch);
      roll = _normalize(roll);
    }

    _lastTimestampUs = now;
  }

  // ------------------------------------------------------------
  // Angle normalization
  // ------------------------------------------------------------

  double _normalize(double angle) {
    while (angle > 180) {angle -= 360;}
    while (angle < -180) {angle += 360;}
    return angle;
  }

  // ------------------------------------------------------------
  // Get rotation matrix values (future VR use)
  // ------------------------------------------------------------

  List<double> get rotationVector => [yaw, pitch, roll];

  @override
  void dispose() {
    _uiTimer?.cancel();
    super.dispose();
  }
}
