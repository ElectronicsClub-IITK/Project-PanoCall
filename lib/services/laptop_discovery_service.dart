import 'dart:async';
import 'dart:convert';
import 'dart:io';

class DiscoveredLaptop {
  final String name;
  final String ip;
  final int orientationPort;
  final int videoPort;
  DateTime lastSeen;

  DiscoveredLaptop({
    required this.name,
    required this.ip,
    required this.orientationPort,
    required this.videoPort,
    required this.lastSeen,
  });
}

/// Listens for the UDP broadcast beacon laptop_server.py sends out every
/// second, so the app can find a laptop on the local network automatically
/// instead of requiring a hardcoded or manually-typed IP address.
class LaptopDiscoveryService {
  static const int discoveryPort = 9003;
  static const Duration staleTimeout = Duration(seconds: 5);

  RawDatagramSocket? _socket;
  Timer? _cleanupTimer;

  final _laptopsController = StreamController<List<DiscoveredLaptop>>.broadcast();
  Stream<List<DiscoveredLaptop>> get laptops => _laptopsController.stream;

  final Map<String, DiscoveredLaptop> _seen = {};

  Future<void> start() async {
    _socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, discoveryPort);
    _socket!.broadcastEnabled = true;

    _socket!.listen((event) {
      if (event != RawSocketEvent.read) return;
      final datagram = _socket!.receive();
      if (datagram == null) return;

      try {
        final msg = jsonDecode(utf8.decode(datagram.data));
        if (msg["service"] != "imu_vr_laptop") return;

        final ip = (msg["ip"] as String?) ?? datagram.address.address;
        _seen[ip] = DiscoveredLaptop(
          name: msg["name"] ?? "Unknown laptop",
          ip: ip,
          orientationPort: msg["orientation_port"] ?? 9001,
          videoPort: msg["video_port"] ?? 9002,
          lastSeen: DateTime.now(),
        );
        _emit();
      } catch (_) {
        // Ignore malformed packets — could be unrelated traffic on this port.
      }
    });

    // Drop laptops we haven't heard from recently (e.g. it was closed).
    _cleanupTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      final now = DateTime.now();
      final before = _seen.length;
      _seen.removeWhere((_, l) => now.difference(l.lastSeen) > staleTimeout);
      if (_seen.length != before) _emit();
    });
  }

  void _emit() {
    _laptopsController.add(_seen.values.toList());
  }

  void stop() {
    _cleanupTimer?.cancel();
    _socket?.close();
  }

  void dispose() {
    stop();
    _laptopsController.close();
  }
}
