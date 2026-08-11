import 'package:flutter/material.dart';

import '../services/laptop_discovery_service.dart';
import '../services/orientation_service.dart';
import 'vr_view_screen.dart';

/// Shown when you tap "Launch VR View" — listens for laptops on the local
/// network and lets you tap one to connect. No IP address needs to be
/// typed in or hardcoded; this is what makes the app work for anyone
/// with the laptop server running, not just one specific machine.
class DiscoverLaptopScreen extends StatefulWidget {
  final OrientationService orientation;

  const DiscoverLaptopScreen({super.key, required this.orientation});

  @override
  State<DiscoverLaptopScreen> createState() => _DiscoverLaptopScreenState();
}

class _DiscoverLaptopScreenState extends State<DiscoverLaptopScreen> {
  final LaptopDiscoveryService _discovery = LaptopDiscoveryService();
  List<DiscoveredLaptop> _found = [];

  final _manualIpController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _discovery.start();
    _discovery.laptops.listen((list) => setState(() => _found = list));
  }

  @override
  void dispose() {
    _discovery.dispose();
    _manualIpController.dispose();
    super.dispose();
  }

  void _connectTo(String ip) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => VrViewScreen(
          orientation: widget.orientation,
          laptopHost: ip,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Find VR Laptop")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_found.isEmpty) ...[
              const Row(
                children: [
                  SizedBox(
                    width: 20, height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  ),
                  SizedBox(width: 12),
                  Text("Searching for laptops on this WiFi network..."),
                ],
              ),
              const SizedBox(height: 8),
              const Text(
                "Make sure laptop_server.py is running and both devices "
                "are on the same WiFi network.",
                style: TextStyle(color: Colors.grey),
              ),
            ] else ...[
              Text("Found ${_found.length} laptop(s):",
                  style: const TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 8),
              ..._found.map((laptop) => Card(
                    child: ListTile(
                      leading: const Icon(Icons.laptop_mac),
                      title: Text(laptop.name),
                      subtitle: Text(laptop.ip),
                      trailing: const Icon(Icons.arrow_forward_ios, size: 16),
                      onTap: () => _connectTo(laptop.ip),
                    ),
                  )),
            ],

            const Spacer(),
            const Divider(),
            const Text(
              "Not showing up? Enter the laptop's IP manually "
              "(some networks block auto-discovery):",
              style: TextStyle(color: Colors.grey, fontSize: 13),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _manualIpController,
                    decoration: const InputDecoration(
                      hintText: "e.g. 192.168.1.42",
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                    keyboardType: TextInputType.number,
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton(
                  onPressed: () {
                    final ip = _manualIpController.text.trim();
                    if (ip.isNotEmpty) _connectTo(ip);
                  },
                  child: const Text("Connect"),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
