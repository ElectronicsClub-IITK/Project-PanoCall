import 'package:flutter/material.dart';
import '../services/native_sensor_service.dart';

import '../services/imu_service.dart';
import '../services/performance_service.dart';
import '../services/orientation_service.dart';

import '../widgets/performance_card.dart';
import '../widgets/sensor_card.dart';
import 'discover_laptop_screen.dart';
import 'debug_log_screen.dart';

class HomeScreen extends StatefulWidget {
  final ImuService imuService;
  final PerformanceService performance;
  final OrientationService orientation;

  const HomeScreen({
    super.key,
    required this.imuService,
    required this.performance,
    required this.orientation,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {

  final NativeSensorService nativeSensorService = NativeSensorService();

  @override
  void initState() {
    super.initState();

    nativeSensorService.testConnection().then((message) {
      print(message);
    });

    // start() is async now: it tells the native side what output rate
    // to run at (default 200Hz, see ImuService.targetRateHz) before
    // subscribing to the stream.
    widget.imuService.start();
  }

  @override
  void dispose() {
    widget.imuService.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(
      appBar: AppBar(
        title: const Text("VR IMU Monitor"),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.bug_report_outlined),
            tooltip: "Debug log",
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const DebugLogScreen()),
              );
            },
          ),
        ],
      ),

      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.vrpano),
        label: const Text("Launch VR View"),
        onPressed: () {
          Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => DiscoverLaptopScreen(orientation: widget.orientation),
            ),
          );
        },
      ),

      body: AnimatedBuilder(
        animation: Listenable.merge([
          widget.performance,
          widget.orientation,
        ]),

        builder: (context, child) {

          return SingleChildScrollView(
            child: Column(
              children: [

                //------------------------------------------------------------------
                // Performance
                //------------------------------------------------------------------

                PerformanceCard(
                  updateRate: widget.performance.updateRate,
                  averageInterval: widget.performance.averageInterval,
                  minimumInterval: widget.performance.minimumInterval == double.infinity
                      ? 0
                      : widget.performance.minimumInterval,
                  maximumInterval: widget.performance.maximumInterval,
                  jitter: widget.performance.jitter,
                ),

                const SizedBox(height: 10),

                //------------------------------------------------------------------
                // VR Orientation
                //------------------------------------------------------------------

                Card(
                  margin: const EdgeInsets.all(16),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [

                        const Text(
                          "VR Orientation",
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),

                        const SizedBox(height: 10),

                        Text("Yaw   : ${widget.orientation.yaw.toStringAsFixed(2)}"),
                        Text("Pitch : ${widget.orientation.pitch.toStringAsFixed(2)}"),
                        Text("Roll  : ${widget.orientation.roll.toStringAsFixed(2)}"),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 10),

                //------------------------------------------------------------------
                // Accelerometer
                //------------------------------------------------------------------

                SensorCard(
                  title: "Accelerometer",
                  x: widget.imuService.imu.ax,
                  y: widget.imuService.imu.ay,
                  z: widget.imuService.imu.az,
                  rateHz: widget.performance.updateRate,
                  isActive: widget.performance.hasSamples,
                ),

                //------------------------------------------------------------------
                // Gyroscope
                //------------------------------------------------------------------

                SensorCard(
                  title: "Gyroscope",
                  x: widget.imuService.imu.gx,
                  y: widget.imuService.imu.gy,
                  z: widget.imuService.imu.gz,
                  rateHz: widget.performance.updateRate,
                  isActive: widget.performance.hasSamples,
                ),

                //------------------------------------------------------------------
                // Magnetometer
                //------------------------------------------------------------------

                SensorCard(
                  title: "Magnetometer",
                  x: widget.imuService.imu.mx,
                  y: widget.imuService.imu.my,
                  z: widget.imuService.imu.mz,
                  rateHz: widget.performance.updateRate,
                  isActive: widget.performance.hasSamples,
                ),

                const SizedBox(height: 20),
              ],
            ),
          );
        },
      ),
    );
  }
}
