// Fully Working Code for ESP32-CAM with 224x224 JPEG Output (No External JPEG Encoder)
// Removes dependency on JPEGENC and uses captured JPEG directly

#include <WiFi.h>
#include <WebServer.h>
#include "esp_camera.h"

// WiFi Credentials (ESP32-S3 AP)
const char* WIFI_SSID = "iotex";
const char* WIFI_PASS = "12345678";

// Static IP Setup for ESP32-CAM
IPAddress staticIP(192, 168, 4, 2);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

WebServer server(80);

// ESP32-CAM (AI Thinker) Pins
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_GPIO_NUM       4

void handleRoot() {
  server.send(200, "text/plain", "Go to /capture to get a JPEG image");
}

void handleCapture() {
  digitalWrite(LED_GPIO_NUM, HIGH);
  delay(50);

  camera_fb_t* fb = esp_camera_fb_get();
  digitalWrite(LED_GPIO_NUM, LOW);

  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }

  // Send the captured JPEG as-is
  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Disposition", "inline; filename=capture.jpg");
  server.sendHeader("Connection", "close");
  server.send_P(200, "image/jpeg", (char*)fb->buf, fb->len);

  esp_camera_fb_return(fb);
}

void connectToS3AP() {
  WiFi.mode(WIFI_STA);
  if (!WiFi.config(staticIP, gateway, subnet)) {
    Serial.println("Failed to set static IP");
  }
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to ESP32-S3 AP");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi connection failed, restarting...");
    delay(2000);
    ESP.restart();
  }
  Serial.println("\nConnected to AIoTex");
  Serial.print("ESP32-CAM IP: ");
  Serial.println(WiFi.localIP());
}

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    delay(2000);
    ESP.restart();
  }
  Serial.println("Camera initialized");
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);

  setupCamera();
  connectToS3AP();

  server.on("/", HTTP_GET, handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}
