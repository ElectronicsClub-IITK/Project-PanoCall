import 'package:flutter/services.dart';

import '../models/imu_data.dart';

class NativeSensorService {
  static const MethodChannel _methodChannel = MethodChannel('imu_vr/native');
  static const EventChannel _eventChannel = EventChannel('imu_vr/native_stream');

  Future<String> testConnection() async {
    final String message = await _methodChannel.invokeMethod("testConnection");
    return message;
  }

  /// Change the fixed output rate the native side pumps data at.
  /// e.g. setOutputRate(200) for 200Hz, setOutputRate(150) for 150Hz.
  Future<void> setOutputRate(int hz) async {
    await _methodChannel.invokeMethod("setOutputRateHz", {"hz": hz});
  }

  /// One fused accel+gyro+mag packet per event, at whatever rate the
  /// native side is configured to pump (default 200Hz).
  Stream<ImuData> get imuStream {
    return _eventChannel.receiveBroadcastStream().map((event) {
      final map = Map<String, dynamic>.from(event as Map);
      return ImuData(
        ax: (map['ax'] as num).toDouble(),
        ay: (map['ay'] as num).toDouble(),
        az: (map['az'] as num).toDouble(),
        gx: (map['gx'] as num).toDouble(),
        gy: (map['gy'] as num).toDouble(),
        gz: (map['gz'] as num).toDouble(),
        mx: (map['mx'] as num).toDouble(),
        my: (map['my'] as num).toDouble(),
        mz: (map['mz'] as num).toDouble(),
        timestamp: (map['timestamp'] as num).toInt(),
      );
    });
  }
}
