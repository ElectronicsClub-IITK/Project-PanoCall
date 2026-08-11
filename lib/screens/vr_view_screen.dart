import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/laptop_link_service.dart';
import '../services/orientation_service.dart';

/// Full-screen VR viewer. Connects to the laptop, sends this phone's
/// orientation, and displays whatever stereo frame comes back —
/// the laptop already renders left+right eye side by side, so this
/// screen just shows the frame edge-to-edge.
class VrViewScreen extends StatefulWidget {
  final OrientationService orientation;

  /// Laptop's address. "127.0.0.1" if using USB + adb reverse,
  /// or the laptop's WiFi LAN IP (e.g. "192.168.1.42") if using WiFi.
  final String laptopHost;

  const VrViewScreen({
    super.key,
    required this.orientation,
    this.laptopHost = "127.0.0.1",
  });

  @override
  State<VrViewScreen> createState() => _VrViewScreenState();
}

class _VrViewScreenState extends State<VrViewScreen> {
  late final LaptopLinkService _link;

  Uint8List? _latestFrame;
  String _status = "Connecting...";

  @override
  void initState() {
    super.initState();

    _link = LaptopLinkService(host: widget.laptopHost);

    // Go full-screen and landscape — this is a VR screen, not a UI screen.
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);

    _link.status.listen((s) => setState(() => _status = s));
    _link.frames.listen((frame) => setState(() => _latestFrame = frame));

    _link.connect(widget.orientation).catchError((_) {
      // status stream already reports the failure; nothing else to do here
    });
  }

  @override
  void dispose() {
    _link.dispose();
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    SystemChrome.setPreferredOrientations([]);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            if (_latestFrame != null)
              Positioned.fill(
                child: Image.memory(
                  _latestFrame!,
                  fit: BoxFit.contain,
                  gaplessPlayback: true,
                ),
              )
            else
              Center(
                child: Text(
                  _status,
                  style: const TextStyle(color: Colors.white, fontSize: 18),
                ),
              ),

            // Small exit button — tap to leave VR mode.
            Positioned(
              top: 8,
              left: 8,
              child: SafeArea(
                child: IconButton(
                  icon: const Icon(Icons.close, color: Colors.white70),
                  onPressed: () => Navigator.of(context).pop(),
                ),
              ),
            ),

            // Recenter / calibrate button — zeroes yaw/pitch/roll to
            // wherever you're currently facing. Use this any time drift
            // has built up or you just want "forward" to mean "here".
            Positioned(
              top: 8,
              right: 8,
              child: SafeArea(
                child: IconButton(
                  icon: const Icon(Icons.center_focus_strong, color: Colors.white70),
                  tooltip: "Recenter view",
                  onPressed: () => widget.orientation.reset(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
