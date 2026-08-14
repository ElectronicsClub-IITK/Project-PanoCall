import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Shows the native side's error/diagnostic log, right inside the app.
/// Needed because Android 11+ blocks file manager apps from browsing
/// into Android/data/<package>/ — this reads the log directly instead,
/// no adb, no developer mode, no file manager required.
class DebugLogScreen extends StatefulWidget {
  const DebugLogScreen({super.key});

  @override
  State<DebugLogScreen> createState() => _DebugLogScreenState();
}

class _DebugLogScreenState extends State<DebugLogScreen> {
  static const MethodChannel _channel = MethodChannel('imu_vr/native');

  String _log = "Loading...";

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    try {
      final result = await _channel.invokeMethod("readLog");
      setState(() => _log = result as String);
    } catch (e) {
      setState(() => _log = "Failed to read log: $e");
    }
  }

  Future<void> _clear() async {
    try {
      await _channel.invokeMethod("clearLog");
      await _refresh();
    } catch (e) {
      setState(() => _log = "Failed to clear log: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    // Highlight likely-important lines so you don't have to read every word.
    final lines = _log.split('\n');

    return Scaffold(
      appBar: AppBar(
        title: const Text("Debug Log"),
        actions: [
          IconButton(icon: const Icon(Icons.refresh), onPressed: _refresh),
          IconButton(icon: const Icon(Icons.delete_outline), onPressed: _clear),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(12),
        child: SelectableText.rich(
          TextSpan(
            children: lines.map((line) {
              final isImportant = line.contains("Exception") ||
                  line.contains("Error") ||
                  line.contains("failed") ||
                  line.contains("granted: false");
              return TextSpan(
                text: "$line\n",
                style: TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: isImportant ? Colors.redAccent : Colors.black87,
                  fontWeight: isImportant ? FontWeight.bold : FontWeight.normal,
                ),
              );
            }).toList(),
          ),
        ),
      ),
    );
  }
}
