import 'package:flutter/material.dart';


class SensorCard extends StatelessWidget {
  final String title;

  final double x;
  final double y;
  final double z;

  final double? rateHz;
  final bool? isActive;

  const SensorCard({
    super.key,
    required this.title,
    required this.x,
    required this.y,
    required this.z,
    this.rateHz,
    this.isActive,
  });

  Color getStatusColor() {
    if (isActive == false) return Colors.red;
    if (rateHz == null) return Colors.grey;

    if (rateHz! >= 90) return Colors.green;
    if (rateHz! >= 60) return Colors.orange;
    return Colors.red;
  }

  Widget valueRow(String label, double value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          Text(
            label,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const Spacer(),
          Text(
            value.toStringAsFixed(3),
            style: const TextStyle(
              fontFamily: 'monospace',
              fontSize: 16,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Row(
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),

                // Status indicator dot
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: getStatusColor(),
                    shape: BoxShape.circle,
                  ),
                ),
              ],
            ),

            const SizedBox(height: 8),

            // Optional rate display
            if (rateHz != null)
              Text(
                "${rateHz!.toStringAsFixed(1)} Hz",
                style: const TextStyle(
                  fontSize: 14,
                  color: Colors.grey,
                ),
              ),

            const Divider(),

            // Axis values
            valueRow("X", x),
            valueRow("Y", y),
            valueRow("Z", z),
          ],
        ),
      ),
    );
  }
}