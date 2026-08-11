import 'package:flutter/material.dart';

import 'screens/home_screen.dart';
import 'services/imu_service.dart';
import 'services/performance_service.dart';
import 'services/orientation_service.dart';

void main() {
  final performance = PerformanceService();
  final orientation = OrientationService();

  final imuService = ImuService(
    performance: performance,
    orientation: orientation,
  );

  runApp(ImuVrApp(
    imuService: imuService,
    performance: performance,
    orientation: orientation,
  ));
}

class ImuVrApp extends StatelessWidget {
  final ImuService imuService;
  final PerformanceService performance;
  final OrientationService orientation;

  const ImuVrApp({
    super.key,
    required this.imuService,
    required this.performance,
    required this.orientation,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen(
        imuService: imuService,
        performance: performance,
        orientation: orientation,
      ),
    );
  }
}     