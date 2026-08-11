import 'dart:async';
import 'package:flutter/foundation.dart';

class PerformanceService extends ChangeNotifier {
  //--------------------------------------------------------------------------
  // Configuration
  //--------------------------------------------------------------------------

  static const int uiRefreshRate = 30; // FPS — UI notify rate, decoupled from sample rate

  //--------------------------------------------------------------------------
  // Timer
  //--------------------------------------------------------------------------

  final Stopwatch _stopwatch = Stopwatch()..start();

  int _lastTimestampUs = 0;

  Timer? _uiTimer;

  PerformanceService() {
    // addSample() can now be called at 150-200Hz. We don't want to
    // notifyListeners() that often (causes UI jank), so a fixed timer
    // pushes UI updates at uiRefreshRate while stats accumulate at
    // full sensor rate underneath.
    _uiTimer = Timer.periodic(
      Duration(milliseconds: (1000 / uiRefreshRate).round()),
      (_) => notifyListeners(),
    );
  }

  //--------------------------------------------------------------------------
  // Statistics
  //--------------------------------------------------------------------------

  int sampleCount = 0;

  double currentInterval = 0.0;
  double averageInterval = 0.0;
  double minimumInterval = double.infinity;
  double maximumInterval = 0.0;

  double updateRate = 0.0;
  double jitter = 0.0;

  double _intervalSum = 0.0;
  double _previousInterval = 0.0;

  //--------------------------------------------------------------------------
  // Called whenever a NEW sensor sample arrives (now potentially 150-200Hz)
  //--------------------------------------------------------------------------

  void addSample() {
    final now = _stopwatch.elapsedMicroseconds;

    if (_lastTimestampUs != 0) {
      currentInterval = (now - _lastTimestampUs) / 1000.0; // ms

      sampleCount++;

      _intervalSum += currentInterval;
      averageInterval = _intervalSum / sampleCount;

      // At 200Hz, a healthy interval is ~5ms, so the old ">0.5ms" sanity
      // floor still works fine — keeping it since it filters out
      // duplicate/zero-delta timestamps.
      if (currentInterval > 0.5 && currentInterval < minimumInterval) {
        minimumInterval = currentInterval;
      }

      if (currentInterval > maximumInterval) {
        maximumInterval = currentInterval;
      }

      if (_previousInterval != 0) {
        jitter = (currentInterval - _previousInterval).abs();
      }

      if (averageInterval > 0) {
        updateRate = 1000.0 / averageInterval;
      }

      _previousInterval = currentInterval;
    }

    _lastTimestampUs = now;
  }

  //--------------------------------------------------------------------------
  // Reset all statistics
  //--------------------------------------------------------------------------

  void reset() {
    sampleCount = 0;
    currentInterval = 0;
    averageInterval = 0;
    minimumInterval = double.infinity;
    maximumInterval = 0;
    updateRate = 0;
    jitter = 0;
    _intervalSum = 0;
    _previousInterval = 0;
    _lastTimestampUs = 0;

    _stopwatch
      ..reset()
      ..start();

    notifyListeners();
  }

  //--------------------------------------------------------------------------
  // Sensor Health — thresholds bumped since target is now 150-200Hz
  //--------------------------------------------------------------------------

  bool get hasSamples => sampleCount > 0;

  bool get isHealthy => updateRate >= 150;

  String get healthStatus {
    if (updateRate >= 150) return "Excellent";
    if (updateRate >= 90) return "Good";
    if (updateRate >= 50) return "Fair";
    return "Poor";
  }

  //--------------------------------------------------------------------------
  // Stability (0 - 100)
  //--------------------------------------------------------------------------

  double get stabilityScore {
    if (averageInterval == 0) return 100;
    final score = 100 - ((jitter / averageInterval) * 100);
    return score.clamp(0, 100);
  }

  //--------------------------------------------------------------------------
  // Future placeholders
  //--------------------------------------------------------------------------

  double usbLatency = 0;
  double pcProcessingLatency = 0;
  double renderLatency = 0;
  double endToEndLatency = 0;

  @override
  void dispose() {
    _uiTimer?.cancel();
    super.dispose();
  }
}
