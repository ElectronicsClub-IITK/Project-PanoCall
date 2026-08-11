class ImuData {
  // Accelerometer (m/s²)
  double ax;
  double ay;
  double az;

  // Gyroscope (rad/s)
  double gx;
  double gy;
  double gz;

  // Magnetometer (µT)
  double mx;
  double my;
  double mz;

  // Timestamp (microseconds)
  int timestamp;

  ImuData({
    this.ax = 0,
    this.ay = 0,
    this.az = 0,
    this.gx = 0,
    this.gy = 0,
    this.gz = 0,
    this.mx = 0,
    this.my = 0,
    this.mz = 0,
    this.timestamp = 0,
  });

  Map<String, dynamic> toJson() {
    return {
      "timestamp": timestamp,

      "accelerometer": {
        "x": ax,
        "y": ay,
        "z": az,
      },

      "gyroscope": {
        "x": gx,
        "y": gy,
        "z": gz,
      },

      "magnetometer": {
        "x": mx,
        "y": my,
        "z": mz,
      },
    };
  }
}