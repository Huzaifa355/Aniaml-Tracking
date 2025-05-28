#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <JPEGDecoder.h>
#include "esp_heap_caps.h"
#include <algorithm>
#include <time.h>

// TensorFlow Lite includes
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "animal_model_data.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

#define IR_A 4
#define IR_B 5

const char* ssid     = "Dhanju";
const char* password = "Huzaifa355";
const char* cam_ip   = "http://192.168.100.110/capture";

// InfluxDB config - replace with your actual InfluxDB info
const char* influx_host   = "http://192.168.100.102:8086";
const char* influx_org    = "student";
const char* influx_bucket = "Animal_Tracking";
const char* influx_token  = "aQSYB7v7Jhz3d2FX03G7MSDq4QYecPoTe1hyxvd5Bw1XnZkEVVK7Gde8ivZUuAcry1PDQro244dYhlOWZ-5ttw==";

constexpr int kTensorArenaSize = 4 * 1024 * 1024;
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

int camelCount = 0;
int cowCount = 0;
int goatCount = 0;

void setupWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.println("\nConnected: " + WiFi.localIP().toString());
}

void initModel() {
  const tflite::Model* model = tflite::GetModel(animal_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  if (!psramFound() || ESP.getFreePsram() < kTensorArenaSize) {
    Serial.println("Insufficient PSRAM!");
    return;
  }

  tensor_arena = static_cast<uint8_t*>(heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM));
  if (!tensor_arena) {
    Serial.println("Tensor arena allocation failed!");
    return;
  }

  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddMean();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);
}

void updateOLED() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.printf("Camels: %d\nCows: %d\nGoats: %d\nTotal: %d",
                camelCount, cowCount, goatCount,
                camelCount + cowCount + goatCount);
  display.display();
}

// Publish data to InfluxDB using HTTP POST
void publishInfluxDB(int label, const String& direction) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, can't send to InfluxDB");
    return;
  }

  time_t now = time(nullptr);

  String animal;
  if (label == 0) animal = "camel";
  else if (label == 1) animal = "cow";
  else animal = "goat";

  // Format timestamp for InfluxDB line protocol (nanoseconds precision)
  // Multiply seconds by 1,000,000,000 to get nanoseconds
  char timeStr[30];
  snprintf(timeStr, sizeof(timeStr), "%lld", (int64_t)now * 1000000000LL);

  // Prepare line protocol: measurement,tag=value field=value timestamp
  // Example: animal_tracker,animal=camel,direction=IN value=1 1591601820000000000
  String data = "animal_tracker";
  data += ",animal=" + animal;
  data += ",direction=" + direction;
  data += " value=1 ";
  data += timeStr;

  HTTPClient http;
  String url = String(influx_host) + "/api/v2/write?org=" + influx_org + "&bucket=" + influx_bucket + "&precision=ns";

  http.begin(url);
  http.addHeader("Authorization", "Token " + String(influx_token));
  http.addHeader("Content-Type", "text/plain");

  int httpResponseCode = http.POST(data);

  if (httpResponseCode > 0) {
    Serial.printf("InfluxDB write successful, code: %d\n", httpResponseCode);
  } else {
    Serial.printf("InfluxDB write failed, error: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  http.end();
}

void classifyAndUpdate(const String& direction) {
  HTTPClient http;
  http.begin(cam_ip);
  int httpCode = http.GET();

  if (httpCode == HTTP_CODE_OK) {
    WiFiClient* stream = http.getStreamPtr();
    File temp = SPIFFS.open("/tmp.jpg", FILE_WRITE);
    if (!temp) {
      Serial.println("Temp file fail");
      http.end();
      return;
    }

    uint8_t buffer[512];
    size_t totalBytes = 0;
    while (stream->available()) {
      size_t bytesRead = stream->readBytes(buffer, sizeof(buffer));
      temp.write(buffer, bytesRead);
      totalBytes += bytesRead;
    }
    temp.close();

    if (!SPIFFS.exists("/tmp.jpg") || totalBytes == 0) {
      Serial.println("File write failed");
      http.end();
      return;
    }

    bool ok = JpegDec.decodeFsFile("/tmp.jpg");
    if (ok && JpegDec.width >= 128 && JpegDec.height >= 128) {
      int w = JpegDec.width, h = JpegDec.height;
      for (int y = 0; y < 128; y++) {
        for (int x = 0; x < 128; x++) {
          int sx = map(x, 0, 127, 0, w - 1);
          int sy = map(y, 0, 127, 0, h - 1);
          uint16_t p = JpegDec.pImage[sy * w + sx];
          uint8_t r = ((p >> 11) & 0x1F) * 255 / 31;
          uint8_t g = ((p >> 5) & 0x3F) * 255 / 63;
          uint8_t b = (p & 0x1F) * 255 / 31;

          int pixel_index = (y * 128 + x) * 3;
          input->data.uint8[pixel_index + 0] = r;
          input->data.uint8[pixel_index + 1] = g;
          input->data.uint8[pixel_index + 2] = b;
        }
      }

      if (interpreter->Invoke() == kTfLiteOk) {
        int best = 0;
        float maxVal = output->data.f[0];
        for (int i = 1; i < output->dims->data[1]; ++i) {
          if (output->data.f[i] > maxVal) {
            maxVal = output->data.f[i];
            best = i;
          }
        }

        const float confidenceThreshold = 0.7f;  // Confidence threshold to avoid false positives
        if (maxVal < confidenceThreshold) {
          Serial.println("Low confidence, ignoring detection");
          JpegDec.abort();
          SPIFFS.remove("/tmp.jpg");
          http.end();
          return;
        }

        if (direction == "IN") {
          if (best == 0) camelCount++;
          else if (best == 1) cowCount++;
          else goatCount++;
        } else {
          if (best == 0) camelCount = std::max(0, camelCount - 1);
          else if (best == 1) cowCount = std::max(0, cowCount - 1);
          else goatCount = std::max(0, goatCount - 1);
        }

        updateOLED();
        publishInfluxDB(best, direction);
      }
    } else {
      Serial.println(ok ? "Image too small" : "JPEG decode failed");
    }

    JpegDec.abort();
    SPIFFS.remove("/tmp.jpg");
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  if (!psramFound()) {
    Serial.println("No PSRAM available!");
    while(1);
  }

  pinMode(IR_A, INPUT);
  pinMode(IR_B, INPUT);

  Wire.begin(9, 8);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED init failed");
    while(1);
  }

  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS mount failed");
    while(1);
  }

  setupWiFi();
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");  // Initialize time for timestamps
  initModel();
  updateOLED();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Reconnecting WiFi");
    setupWiFi();
  }

  static bool lastA = digitalRead(IR_A);
  static bool lastB = digitalRead(IR_B);

  bool currentA = digitalRead(IR_A);
  bool currentB = digitalRead(IR_B);

  if (lastA != currentA || lastB != currentB) {
    if (currentA && !currentB) classifyAndUpdate("IN");
    else if (!currentA && currentB) classifyAndUpdate("OUT");

    lastA = currentA;
    lastB = currentB;
    delay(50);  // small debounce delay
  }
}
