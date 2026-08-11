import 'package:flutter/material.dart';

class PerformanceCard extends StatelessWidget {
  final double updateRate;
  final double averageInterval;
  final double minimumInterval;
  final double maximumInterval;
  final double jitter;

  const PerformanceCard({
    super.key,
    required this.updateRate,
    required this.averageInterval,
    required this.minimumInterval,
    required this.maximumInterval,
    required this.jitter,
  });

  Color get statusColor {
    if (updateRate >= 90) {
      return Colors.green;
    } else if (updateRate >= 60) {
      return Colors.orange;
    } else {
      return Colors.red;
    }
  }

  String get statusText {
    if (updateRate >= 90) {
      return "Excellent";
    } else if (updateRate >= 60) {
      return "Good";
    } else {
      return "Low";
    }
  }

  Widget metric(String title, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        children: [

          Expanded(
            child: Text(
              title,
              style: const TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 16,
              ),
            ),
          ),

          Text(
            value,
            style: const TextStyle(
              fontSize: 16,
              fontFamily: 'monospace',
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {

    return Card(

      elevation: 5,

      margin: const EdgeInsets.all(16),

      child: Padding(

        padding: const EdgeInsets.all(16),

        child: Column(

          crossAxisAlignment: CrossAxisAlignment.start,

          children: [

            const Text(
              "VR IMU Performance",
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 12),

            Row(
              children: [

                const Text(
                  "Status : ",
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),

                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: statusColor,
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    statusText,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),

              ],
            ),

            const Divider(height: 25),

            metric(
              "Update Rate",
              "${updateRate.toStringAsFixed(1)} Hz",
            ),

            metric(
              "Average Interval",
              "${averageInterval.toStringAsFixed(2)} ms",
            ),

            metric(
              "Minimum Interval",
              "${minimumInterval.toStringAsFixed(2)} ms",
            ),

            metric(
              "Maximum Interval",
              "${maximumInterval.toStringAsFixed(2)} ms",
            ),

            metric(
              "Jitter",
              "${jitter.toStringAsFixed(2)} ms",
            ),

          

          ],

        ),

      ),

    );

  }
}