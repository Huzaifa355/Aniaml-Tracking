#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <JPEGDecoder.h>
#include <time.h>
#include "esp_task_wdt.h"

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

// Wi-Fi credentials
const char* ssid     = "Dhanju";
const char* password = "Huzaifa355";

// Camera endpoint
const char* cam_ip   = "http://192.168.100.110/capture";

// InfluxDB config
const char* influx_host   = "http://192.168.137.1:8086";
const char* influx_org    = "student";
const char* influx_bucket = "Animal_Tracking";
const char* influx_token  = "aCBR21LJQE_S2nhPNGwEucEXjHWo0fdbXVhwOUNlYUAkXXTTc1jFbaNfwrUghdxQc_ALuenbe30hzPD7vAfSqw==";

// Updated tensor arena size for 224x224 model
constexpr int kTensorArenaSize = 6 * 1024 * 1024;  // Increased to 6MB
uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// Model input dimensions - Updated to match your model
const int IMG_WIDTH = 224;   // Changed from 96 to 224
const int IMG_HEIGHT = 224;  // Changed from 96 to 224
const int IMG_CHANNELS = 3;  // RGB

// Counts
int cowCount   = 0;
int goatCount  = 0;
int henCount   = 0;
int totalCount = 0;

// Last detection confidence
float lastConfidence = 0.0f;

// Forward declarations
void setupWiFi();
void initModel();
void showWelcome();
void fetchLastCounts();
void updateOLED();
void publishInfluxDB(int label, const String& direction);
void classifyAndUpdate(const String& direction);

void softmax(float* input, int length) {
  float maxVal = input[0];
  for (int i = 1; i < length; i++) {
    if (input[i] > maxVal) maxVal = input[i];
  }

  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    input[i] = expf(input[i] - maxVal);
    sum += input[i];
  }

  for (int i = 0; i < length; i++) {
    input[i] /= sum;
  }
}

// Improved bilinear interpolation for better image quality
void resizeImageBilinear(
  const uint8_t* fullImage, int srcWidth, int srcHeight,
  float* output, int dstWidth, int dstHeight)
{
  float x_ratio = (float)(srcWidth - 1) / dstWidth;
  float y_ratio = (float)(srcHeight - 1) / dstHeight;

  for (int y = 0; y < dstHeight; y++) {
    float srcY = y * y_ratio;
    int yFloor = (int)srcY;
    int yCeil = min(yFloor + 1, srcHeight - 1);
    float yWeight = srcY - yFloor;

    for (int x = 0; x < dstWidth; x++) {
      float srcX = x * x_ratio;
      int xFloor = (int)srcX;
      int xCeil = min(xFloor + 1, srcWidth - 1);
      float xWeight = srcX - xFloor;

      for (int c = 0; c < 3; c++) {
        // Get four neighboring pixels
        float topLeft     = fullImage[(yFloor * srcWidth + xFloor) * 3 + c];
        float topRight    = fullImage[(yFloor * srcWidth + xCeil) * 3 + c];
        float bottomLeft  = fullImage[(yCeil * srcWidth + xFloor) * 3 + c];
        float bottomRight = fullImage[(yCeil * srcWidth + xCeil) * 3 + c];

        // Bilinear interpolation
        float top    = topLeft + (topRight - topLeft) * xWeight;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * xWeight;
        float pixel  = top + (bottom - top) * yWeight;

        // Normalize to [-1, 1] range as expected by MobileNet
        int outputIdx = (y * dstWidth + x) * 3 + c;
        output[outputIdx] = (pixel / 127.5f) - 1.0f;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  
  // Increase CPU frequency for better performance
  Serial.println("Setting CPU frequency to 240MHz");
  setCpuFrequencyMhz(240);
  pinMode(2, OUTPUT);
  digitalWrite(2, HIGH);

  // PSRAM check - Critical for 224x224 processing
  if (!psramFound()) {
    Serial.println("ERROR: No PSRAM available! This model requires PSRAM.");
    while(1);
  }
  
  size_t psramSize = ESP.getPsramSize();
  size_t freePsram = ESP.getFreePsram();
  Serial.printf("PSRAM Total: %d bytes, Free: %d bytes\n", psramSize, freePsram);

  // GPIO inits
  pinMode(IR_A, INPUT);
  pinMode(IR_B, INPUT);

  // OLED init
  Wire.begin(9, 8);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED initialization failed");
    while(1);
  }

  // SPIFFS init
  if (!SPIFFS.begin(true)) {
    Serial.println("SPIFFS mount failed");
    while(1);
  }

  // 1) Welcome screen
  showWelcome();

  // 2) Connect Wi-Fi
  setupWiFi();

  // 3) Setup time synchronization
  configTime(0, 0, "pool.ntp.org", "time.nist.gov");

  // 4) Load last saved counts from InfluxDB
  fetchLastCounts();

  // 5) Initialize TensorFlow model
  initModel();

  // 6) Draw initial main screen
  updateOLED();
  
  Serial.println("System ready for animal classification!");
}

void loop() {
  // Keep Wi-Fi alive
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Reconnecting WiFi...");
    setupWiFi();
  }

  static bool lastA = digitalRead(IR_A);
  static bool lastB = digitalRead(IR_B);

  bool currentA = digitalRead(IR_A);
  bool currentB = digitalRead(IR_B);

  // Detect direction based on IR sensor sequence
  if (lastA != currentA || lastB != currentB) {
    if (currentA && !currentB) {
      Serial.println("Motion detected: IN direction");
      classifyAndUpdate("IN");
    }
    else if (!currentA && currentB) {
      Serial.println("Motion detected: OUT direction");
      classifyAndUpdate("OUT");
    }
    
    lastA = currentA;
    lastB = currentB;
    delay(100);  // Debounce delay
  }
  
  delay(10); // Small delay to prevent excessive polling
}

// ———————————————— Helper Functions —————————————————

void showWelcome() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println("Welcome");
  display.setTextSize(1);
  display.setCursor(0, 35);
  display.println("Animal AI Tracker");
  display.setCursor(0, 45);
  display.println("MobileNetV1 224x224");
  display.setCursor(0, 55);
  display.println("Powered by AIotex");
  display.display();
  delay(3000);
}

void setupWiFi() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Connecting to WiFi...");
  display.display();

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print('.');
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected!");
    Serial.println("IP address: " + WiFi.localIP().toString());
    
    display.clearDisplay();
    display.setCursor(0,0);
    display.println("WiFi Connected!");
    display.setCursor(0,10);
    display.println("IP: " + WiFi.localIP().toString());
    display.display();
    delay(2000);
  } else {
    Serial.println("\nWiFi connection failed!");
    display.clearDisplay();
    display.setCursor(0,0);
    display.println("WiFi Failed!");
    display.display();
    delay(2000);
  }
}

void initModel() {
  Serial.println("Initializing TensorFlow Lite model...");
  
  const tflite::Model* model = tflite::GetModel(animal_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model schema version mismatch! Expected %d, got %d\n", 
                  TFLITE_SCHEMA_VERSION, model->version());
    while(1);
  }

  // Allocate tensor arena in PSRAM
  tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena in PSRAM!");
    Serial.printf("Requested size: %d bytes\n", kTensorArenaSize);
    Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
    while(1);
  }
  
  Serial.printf("Tensor arena allocated: %d bytes in PSRAM\n", kTensorArenaSize);

  // Updated op resolver for MobileNetV1 with quantization
  static tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();  // Added for MobileNet
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMean();
  resolver.AddRelu();           // Added for MobileNet activations
  resolver.AddRelu6();          // Added for MobileNet activations

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    Serial.printf("Status: %d\n", allocate_status);
    while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Verify tensor dimensions and types
  if (input->type != kTfLiteFloat32) {
    Serial.printf("ERROR: Expected input type kTfLiteFloat32 (%d), got %d\n", 
                  kTfLiteFloat32, input->type);
    while(1);
  }
  
  if (output->type != kTfLiteFloat32) {
    Serial.printf("ERROR: Expected output type kTfLiteFloat32 (%d), got %d\n", 
                  kTfLiteFloat32, output->type);
    while(1);
  }

  // Verify input dimensions match your model (224x224x3)
  if (input->dims->size != 4 || 
      input->dims->data[1] != 224 || 
      input->dims->data[2] != 224 || 
      input->dims->data[3] != 3) {
    Serial.println("ERROR: Input dimensions don't match expected 224x224x3!");
    Serial.printf("Got dimensions: ");
    for (int i = 0; i < input->dims->size; i++) {
      Serial.printf("%d ", input->dims->data[i]);
    }
    Serial.println();
    while(1);
  }

  // Verify output dimensions (should be [1, 3] for 3 classes)
  if (output->dims->size != 2 || output->dims->data[1] != 3) {
    Serial.println("ERROR: Output dimensions don't match expected [1, 3]!");
    Serial.printf("Got output dimensions: ");
    for (int i = 0; i < output->dims->size; i++) {
      Serial.printf("%d ", output->dims->data[i]);
    }
    Serial.println();
    while(1);
  }

  Serial.println("Model initialized successfully!");
  Serial.printf("Input: [%d, %d, %d, %d] (float32)\n", 
               input->dims->data[0], input->dims->data[1], 
               input->dims->data[2], input->dims->data[3]);
  Serial.printf("Output: [%d, %d] (float32)\n", 
               output->dims->data[0], output->dims->data[1]);
}

void fetchLastCounts() {
  Serial.println("Fetching last counts from InfluxDB...");
  
  const char* fields[3] = { "cowCount", "goatCount", "henCount" };
  int* counts[3]       = { &cowCount, &goatCount, &henCount };

  for (int i = 0; i < 3; ++i) {
    String flux =
      "from(bucket: \"" + String(influx_bucket) + "\")"
      " |> range(start: -30d)"
      " |> filter(fn: (r) => r._measurement == \"animal_counts\" and r._field == \"" 
        + String(fields[i]) + "\")"
      " |> last()";

    String url = String(influx_host) + "/api/v2/query?org=" + influx_org;

    HTTPClient http;
    http.begin(url);
    http.addHeader("Authorization", "Token " + String(influx_token));
    http.addHeader("Content-Type",  "application/vnd.flux");
    http.setTimeout(10000);

    int code = http.POST(flux);
    if (code == 200) {
      String csv = http.getString();
      int nl = csv.lastIndexOf('\n');
      if (nl > 0) {
        String line = csv.substring(nl + 1);
        line.trim();
        if (line.length() > 0) {
          int p1 = line.indexOf(',', 0);
          int p2 = line.indexOf(',', p1+1);
          int p3 = line.indexOf(',', p2+1);
          int p4 = line.indexOf(',', p3+1);
          String val = (p4 > 0) ? line.substring(p3+1, p4) : line.substring(p3+1);
          *counts[i] = val.toInt();
          Serial.printf("Restored %s = %d\n", fields[i], *counts[i]);
        }
      }
    } else {
      Serial.printf("Failed to fetch %s: HTTP %d\n", fields[i], code);
    }
    http.end();
  }

  totalCount = cowCount + goatCount + henCount;
  Serial.printf("Total restored counts → Cows: %d, Goats: %d, Hens: %d, Total: %d\n",
                cowCount, goatCount, henCount, totalCount);
}

void updateOLED() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.printf("Cows:%d Goats:%d Hens:%d", cowCount, goatCount, henCount);
  display.setCursor(0, 12); 
  display.printf("Total Animals: %d", totalCount);
  display.setCursor(0, 24);
  display.printf("Last Confidence: %.2f", lastConfidence);
  display.setCursor(0, 36);
  
  // Show model info
  display.printf("Model: MobileNetV1-224");
  display.setCursor(0, 48);
  display.printf("PSRAM Free: %dKB", ESP.getFreePsram()/1024);
  
  display.display();
}

void publishInfluxDB(int label, const String& direction) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected, skipping InfluxDB write");
    return;
  }

  time_t now = time(nullptr);
  if (now < 1000000000) { // Invalid timestamp
    Serial.println("Invalid timestamp, skipping InfluxDB write");
    return;
  }
  
  char ts[32];
  snprintf(ts, sizeof(ts), "%lld", (int64_t)now * 1000000000LL); // nanoseconds

  static const char* names[3] = { "cow", "goat", "hen" };

  String lp = String("animal_tracker,animal=") + names[label] +
              ",direction=" + direction +
              ",confidence=" + String(lastConfidence, 2) +
              " value=1 " + ts + "\n";

  lp += String("animal_counts cowCount=") + String(cowCount) +
        String(",goatCount=") + String(goatCount) +
        String(",henCount=") + String(henCount) +
        String(",totalCount=") + String(totalCount) +
        " " + ts;

  HTTPClient http;
  String url = String(influx_host) + "/api/v2/write?org=" + influx_org +
              "&bucket=" + influx_bucket + "&precision=ns";
  
  http.begin(url);
  http.addHeader("Authorization", "Token " + String(influx_token));
  http.addHeader("Content-Type", "text/plain");
  http.setTimeout(5000);
  
  int code = http.POST(lp);
  if (code == 204) {
    Serial.println("InfluxDB: Data written successfully");
  } else {
    Serial.printf("InfluxDB write failed, HTTP code: %d\n", code);
    if (code > 0) {
      String response = http.getString();
      Serial.println("Response: " + response);
    }
  }
  http.end();
  updateOLED();
}

void classifyAndUpdate(const String& direction) {
  Serial.println("\n=== Starting Animal Classification ===");
  Serial.printf("Direction: %s\n", direction.c_str());
  Serial.flush();

  unsigned long startTime = millis();

  // Step 1: Capture Image from ESP32-CAM
  Serial.println("[1] Capturing image from camera...");
  HTTPClient http;
  http.begin(cam_ip);
  http.setTimeout(15000); // Increased timeout

  int httpCode = http.GET();
  if (httpCode != HTTP_CODE_OK) {
    Serial.printf("Camera capture failed: HTTP %d\n", httpCode);
    http.end();
    return;
  }

  // Step 2: Read JPEG data
  Serial.println("[2] Reading JPEG data...");
  WiFiClient* stream = http.getStreamPtr();
  size_t jpegSize = http.getSize();
  
  if (jpegSize == 0 || jpegSize > 512000) { // Sanity check
    Serial.printf("Invalid JPEG size: %d bytes\n", jpegSize);
    http.end();
    return;
  }

  uint8_t* jpegData = (uint8_t*)heap_caps_malloc(jpegSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!jpegData) {
    Serial.println("Failed to allocate JPEG buffer");
    http.end();
    return;
  }

  size_t bytesRead = stream->readBytes(jpegData, jpegSize);  
  http.end();

  if (bytesRead != jpegSize) {
    Serial.printf("Incomplete read: %d/%d bytes\n", bytesRead, jpegSize);
    free(jpegData);
    return;
  }

  Serial.printf("JPEG captured: %d bytes\n", bytesRead);

  // Step 3: Decode JPEG
  Serial.println("[3] Decoding JPEG...");
  
  if (!JpegDec.decodeArray(jpegData, bytesRead)) {
    Serial.println("JPEG decode failed");
    free(jpegData);
    JpegDec.abort();
    return;
  }

  uint16_t imgWidth = JpegDec.width;
  uint16_t imgHeight = JpegDec.height;
  Serial.printf("Decoded image: %dx%d pixels\n", imgWidth, imgHeight);

  // Allocate buffer for full RGB image
  size_t rgbSize = imgWidth * imgHeight * 3;
  uint8_t* fullImage = (uint8_t*)heap_caps_malloc(rgbSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!fullImage) {
    Serial.printf("Failed to allocate RGB buffer (%d bytes)\n", rgbSize);
    free(jpegData);
    JpegDec.abort();
    return;
  }
  
  memset(fullImage, 0, rgbSize);

  // Step 4: Convert JPEG MCUs to RGB
  Serial.println("[4] Converting to RGB...");
  
  while (JpegDec.read()) {
    uint16_t* pImg = JpegDec.pImage;
    uint16_t mcuWidth = JpegDec.MCUWidth;
    uint16_t mcuHeight = JpegDec.MCUHeight;
    uint16_t mcuX = JpegDec.MCUx;
    uint16_t mcuY = JpegDec.MCUy;

    for (int y = 0; y < mcuHeight; y++) {
      for (int x = 0; x < mcuWidth; x++) {
        int globalX = mcuX * mcuWidth + x;
        int globalY = mcuY * mcuHeight + y;
        
        if (globalX >= imgWidth || globalY >= imgHeight) continue;

        uint16_t pixel565 = pImg[y * mcuWidth + x];
        
        // Convert RGB565 to RGB888
        uint8_t r = ((pixel565 >> 11) & 0x1F) * 255 / 31;
        uint8_t g = ((pixel565 >> 5) & 0x3F) * 255 / 63;
        uint8_t b = (pixel565 & 0x1F) * 255 / 31;

        int rgbIdx = (globalY * imgWidth + globalX) * 3;
        fullImage[rgbIdx + 0] = r;
        fullImage[rgbIdx + 1] = g;
        fullImage[rgbIdx + 2] = b;
      }
    }
  }

  free(jpegData); // Free JPEG data early
  JpegDec.abort();

  // Step 5: Resize to 224x224 and normalize
  Serial.println("[5] Resizing to 224x224 and normalizing...");
  
  float* inputData = interpreter->typed_input_tensor<float>(0);
  resizeImageBilinear(fullImage, imgWidth, imgHeight, inputData, IMG_WIDTH, IMG_HEIGHT);
  
  free(fullImage); // Free RGB buffer

  // Debug: Check some normalized values
  Serial.println("Sample normalized values:");
  for (int i = 0; i < 9; i++) {
    Serial.printf("  Pixel %d: R=%.3f G=%.3f B=%.3f\n", i,
                  inputData[i*3], inputData[i*3+1], inputData[i*3+2]);
  }

  // Step 6: Run inference
  Serial.println("[6] Running MobileNetV1 inference...");
  
  unsigned long inferenceStart = millis();
  TfLiteStatus invokeStatus = interpreter->Invoke();
  unsigned long inferenceTime = millis() - inferenceStart;
  
  if (invokeStatus != kTfLiteOk) {
    Serial.printf("Inference failed with status: %d\n", invokeStatus);
    return;
  }
  
  Serial.printf("Inference completed in %lu ms\n", inferenceTime);

  // Step 7: Process results
  Serial.println("[7] Processing results...");
  
  float* outputData = interpreter->typed_output_tensor<float>(0);
  const char* labels[] = {"Cow", "Goat", "Hen"};

  // Apply softmax for proper probabilities
  softmax(outputData, 3);

  // Find best prediction
  int bestIdx = 0;
  float bestConf = outputData[0];
  
  Serial.println("Classification results:");
  for (int i = 0; i < 3; i++) {
    Serial.printf("  %s: %.4f (%.1f%%)\n", labels[i], outputData[i], outputData[i] * 100);
    if (outputData[i] > bestConf) {
      bestConf = outputData[i];
      bestIdx = i;
    }
  }

  lastConfidence = bestConf;
  Serial.printf("Best prediction: %s (%.2f confidence)\n", labels[bestIdx], lastConfidence);

  // Step 8: Update counts with confidence threshold
  const float CONFIDENCE_THRESHOLD = 0.6f; // Adjusted for post-quantization model
  
  if (lastConfidence < CONFIDENCE_THRESHOLD) {
    Serial.printf("Confidence %.2f below threshold %.2f - skipping count update\n", 
                  lastConfidence, CONFIDENCE_THRESHOLD);
    updateOLED(); // Still update display to show the detection
    return;
  }

  // Update animal counts
  if (direction == "IN") {
    switch(bestIdx) {
      case 0: cowCount++; break;
      case 1: goatCount++; break;
      case 2: henCount++; break;
    }
    totalCount++;
    Serial.printf("Animal entered: %s (Total: %d)\n", labels[bestIdx], totalCount);
  } 
  else { // OUT
    bool decremented = false;
    switch(bestIdx) {
      case 0: if (cowCount > 0) { cowCount--; decremented = true; } break;
      case 1: if (goatCount > 0) { goatCount--; decremented = true; } break;
      case 2: if (henCount > 0) { henCount--; decremented = true; } break;
    }
    
    if (decremented && totalCount > 0) {
      totalCount--;
      Serial.printf("Animal exited: %s (Total: %d)\n", labels[bestIdx], totalCount);
    }
  }

  // Ensure non-negative counts
  cowCount = max(0, cowCount);
  goatCount = max(0, goatCount);
  henCount = max(0, henCount);
  totalCount = max(0, totalCount);

  // Step 9: Update display and database
  Serial.println("[8] Updating display and database...");
  updateOLED();
  publishInfluxDB(bestIdx, direction);

  unsigned long totalTime = millis() - startTime;
  Serial.printf("=== Classification completed in %lu ms ===\n\n", totalTime);
}