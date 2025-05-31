
---

## Detailed Workflow

1. **Directional Detection**:
   - Two IR sensors placed sequentially detect an animal's direction.
   - Sensor A triggers before Sensor B → Animal entered.
   - Sensor B triggers before Sensor A → Animal exited.

2. **Image Capture & Processing**:
   - ESP32-CAM captures a JPEG image when movement is detected.
   - Built-in LED flash turns on for better lighting during capture.

3. **Image Transfer**:
   - Captured image sent over HTTP or other communication to ESP32-S3.

4. **ML Inference & Classification**:
   - ESP32-S3 decodes JPEG, resizes image, and runs TensorFlow Lite Micro model.
   - Model outputs classification label and confidence score.

5. **Count Update**:
   - Animal count increments or decrements based on direction and classification.

6. **Data Display & Logging**:
   - OLED shows current counts, confidence scores, and timestamp.
   - Data sent to MQTT broker for storage and dashboard visualization.

---

## Hardware Setup

- **ESP32-CAM**:
  - Connect camera pins as per AI-Thinker ESP32-CAM pinout.
  - LED flash connected to GPIO 4.
  - Connect to Wi-Fi network.

- **IR Sensors**:
  - Connect sensor A and B outputs to GPIO pins on ESP32-CAM or ESP32-S3.
  - Ensure sensors detect reliably by adjusting position and thresholds.

- **ESP32-S3**:
  - Connect OLED display via I2C (define SDA, SCL pins).
  - Connect to same Wi-Fi network for MQTT communication.
  - Setup serial communication for debugging.

---

## Software Setup

1. **ESP32-CAM Firmware**:
   - Initialize camera with desired resolution and JPEG quality.
   - Implement IR sensor interrupt or polling for detecting animals.
   - Capture image and serve/send it on detection.

2. **ESP32-S3 Firmware**:
   - Receive images from ESP32-CAM.
   - Decode JPEG, resize and grayscale conversion as needed.
   - Load TensorFlow Lite Micro model.
   - Run inference and classify animal.
   - Update counts and display data on OLED.
   - Publish data to MQTT broker.

3. **Dashboard Setup** (Optional):
   - Setup MQTT broker (e.g., Mosquitto).
   - Configure Telegraf to subscribe MQTT topics and send data to InfluxDB.
   - Build Grafana dashboard for real-time visualization of animal counts.

---

## How to Use

1. Power on ESP32-CAM and ESP32-S3 devices.
2. Ensure Wi-Fi connectivity for both devices.
3. Monitor serial console for camera initialization and Wi-Fi connection status.
4. Animals passing through gate will be detected by IR sensors.
5. ESP32-CAM captures images on detection and sends them to ESP32-S3.
6. ESP32-S3 classifies the animal, updates counts, displays status, and logs data.
7. Access Grafana dashboard to visualize farm animal movement data.

---

## Data Visualization

- Real-time counts per animal type.
- Confidence scores from ML inference.
- Timestamps for each detection event.
- Historical data trends via Grafana.

---

## Troubleshooting

- **Camera initialization fails**: Check wiring and power supply; ensure PSRAM is available for high resolution.
- **Image size too large**: Adjust JPEG quality and frame size in camera config.
- **IR sensors not detecting**: Verify sensor wiring and sensor alignment.
- **Wi-Fi connection issues**: Confirm SSID and password; check network signal strength.
- **ML model accuracy low**: Retrain model with more samples or tune input image preprocessing.
- **OLED not displaying**: Verify I2C pins and library initialization.

---

## Future Enhancements

- Add GPS module for geo-location tagging.
- Implement voice or SMS alerts for unusual activity.
- Extend model for more animal species.
- Use edge AI accelerator hardware for faster inference.
- Add solar power integration for remote deployment.
- Integrate cloud services for remote monitoring and control.

---

## Credits

- Developed by **Huzaifa Shafique**
- Inspired by IoT and Embedded ML projects in smart farming.
- Uses open-source libraries and ESP32 community resources.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please contact:  
**Huzaifa Shafique**  
Email: huzaifashafique355@gmail.com  
GitHub: https://github.com/Huzaifa355
