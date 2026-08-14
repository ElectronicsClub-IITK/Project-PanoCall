import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'orientation_service.dart';

/// Handles both directions of the laptop link:
///  - sends this phone's orientation out over TCP (port 9001)
///  - receives the laptop's rendered stereo frames back (port 9002)
///
/// Both ports are expected to be reachable at 127.0.0.1 because you've
/// run `adb reverse tcp:9001 tcp:9001` and `adb reverse tcp:9002 tcp:9002`
/// on the laptop before starting this — see laptop_server.py's comments.
class LaptopLinkService {
  /// The laptop's address as seen from the phone.
  ///
  /// - Using USB + `adb reverse`: leave this as "127.0.0.1".
  /// - Using WiFi instead (no USB debugging needed): set this to your
  ///   laptop's LAN IP, e.g. "192.168.1.42" — find it on the laptop with
  ///   `ipconfig` (Windows) and look for "IPv4 Address" under your WiFi
  ///   adapter. Both devices must be on the same WiFi network.
  final String host;

  static const int orientationPort = 9001;
  static const int videoPort = 9002;

  Socket? _orientationSocket;
  Socket? _videoSocket;

  Timer? _sendTimer;
  final int sendRateHz;

  final _frameController = StreamController<Uint8List>.broadcast();
  Stream<Uint8List> get frames => _frameController.stream;

  bool _connected = false;
  bool get isConnected => _connected;

  final _statusController = StreamController<String>.broadcast();
  Stream<String> get status => _statusController.stream;

  LaptopLinkService({this.host = "127.0.0.1", this.sendRateHz = 90});

  Future<void> connect(OrientationService orientation) async {
    try {
      _statusController.add("Connecting to laptop...");

      _orientationSocket = await Socket.connect(host, orientationPort,
          timeout: const Duration(seconds: 5));
      _videoSocket = await Socket.connect(host, videoPort,
          timeout: const Duration(seconds: 5));

      _connected = true;
      _statusController.add("Connected");

      // Push orientation at a fixed rate (doesn't need to match the
      // full sensor rate — 60-100Hz is plenty for smooth head tracking
      // over this link).
      _sendTimer = Timer.periodic(
        Duration(milliseconds: (1000 / sendRateHz).round()),
        (_) => _sendOrientation(orientation),
      );

      _listenForFrames();
    } catch (e) {
      _connected = false;
      _statusController.add("Connection failed: $e");
      rethrow;
    }
  }

  void _sendOrientation(OrientationService orientation) {
    final socket = _orientationSocket;
    if (socket == null) return;

    final payload = jsonEncode({
      "yaw": orientation.yaw,
      "pitch": orientation.pitch,
      "roll": orientation.roll,
    });

    try {
      socket.write("$payload\n");
    } catch (e) {
      _statusController.add("Orientation send error: $e");
    }
  }

  // Frame framing: 4-byte big-endian length prefix, then that many
  // JPEG bytes. TCP doesn't preserve message boundaries, so we buffer
  // and parse manually.
  final List<int> _recvBuffer = [];
  int? _expectedLength;

  void _listenForFrames() {
    _videoSocket!.listen(
      (data) {
        _recvBuffer.addAll(data);
        _tryParseFrames();
      },
      onError: (e) => _statusController.add("Video stream error: $e"),
      onDone: () {
        _connected = false;
        _statusController.add("Laptop disconnected");
      },
    );
  }

  void _tryParseFrames() {
    while (true) {
      if (_expectedLength == null) {
        if (_recvBuffer.length < 4) return;
        final header = Uint8List.fromList(_recvBuffer.sublist(0, 4));
        _expectedLength = ByteData.sublistView(header).getUint32(0);
        _recvBuffer.removeRange(0, 4);
      }

      final len = _expectedLength!;
      if (_recvBuffer.length < len) return;

      final frameBytes = Uint8List.fromList(_recvBuffer.sublist(0, len));
      _recvBuffer.removeRange(0, len);
      _expectedLength = null;

      _frameController.add(frameBytes);
    }
  }

  void disconnect() {
    _sendTimer?.cancel();
    _orientationSocket?.destroy();
    _videoSocket?.destroy();
    _connected = false;
  }

  void dispose() {
    disconnect();
    _frameController.close();
    _statusController.close();
  }
}
