  #include <Arduino.h>
  #include <WiFi.h>
  #include <HTTPClient.h>
  #include <SPIFFS.h>
  #include <Adafruit_GFX.h>
  #include <Adafruit_SSD1306.h>
  #include <JPEGDecoder.h>
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

  constexpr int kTensorArenaSize = 4 * 1024 * 1024;
  uint8_t* tensor_arena = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input  = nullptr;
  TfLiteTensor* output = nullptr;

  // Counts
  int cowCount   = 0;
  int goatCount  = 0;
  int henCount   = 0;
  int totalCount = 0;   // ← new

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

  void setup() {
    Serial.begin(115200);

    // PSRAM check
    if (!psramFound()) {
      Serial.println("No PSRAM available!");
      while(1);
    }

    // GPIO inits
    pinMode(IR_A, INPUT);
    pinMode(IR_B, INPUT);

    // OLED init
    Wire.begin(9, 8);
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
      Serial.println("OLED init failed");
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

    // Prepare accurate timestamps
    configTime(0, 0, "pool.ntp.org", "time.nist.gov");

    // 4) Load last saved counts from InfluxDB
    fetchLastCounts();

    // TensorFlow init
    initModel();

    // Draw initial main screen
    updateOLED();
  }

  void loop() {
    // Keep Wi-Fi alive
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
      delay(50);  // debounce
    }
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
    display.println("Powered by AIotex");
    display.display();
    delay(5000);
  }

  void setupWiFi() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Connecting to WiFi...");
    display.display();

    WiFi.begin(ssid, password);
    Serial.print("Connecting WiFi");
    while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print('.');
    }
    Serial.println("\nConnected: " + WiFi.localIP().toString());

    display.clearDisplay();
    display.setCursor(0,0);
    display.println("WiFi connected!");
    display.display();
    delay(1000);
  }

  void initModel() {
    const tflite::Model* model = tflite::GetModel(animal_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model schema mismatch!");
      return;
    }
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);

    static tflite::MicroMutableOpResolver<9> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddMaxPool2D();
    resolver.AddFullyConnected(); resolver.AddSoftmax(); resolver.AddReshape();
    resolver.AddQuantize(); resolver.AddMean(); resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize
    );
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input  = interpreter->input(0);
    output = interpreter->output(0);
  }

  void fetchLastCounts() {
    const char* fields[3] = { "cowCount", "goatCount", "henCount" };
    int* counts[3]       = { &cowCount, &goatCount, &henCount };

    for (int i = 0; i < 3; ++i) {
      String flux =
        "from(bucket: \"" + String(influx_bucket) + "\")"
        " |> range(start: -30d)"
        " |> filter(fn: (r) => r._measurement == \"animal_counts\" and r._field == \"" 
          + String(fields[i]) + "\")"
        " |> last()";

      String url = String(influx_host)
                  + "/api/v2/query?org=" + influx_org;

      HTTPClient http;
      http.begin(url);
      http.addHeader("Authorization", "Token " + String(influx_token));
      http.addHeader("Content-Type",  "application/vnd.flux");

      Serial.printf("Fetching %s → URL: %s\n", fields[i], url.c_str());
      Serial.printf("Auth header: Token %s\n", influx_token);

      int code = http.POST(flux);
      if (code == 200) {
        String csv = http.getString();
        int nl = csv.lastIndexOf('\n');
        String line = csv.substring(nl + 1);
        line.trim();
        if (line.length()) {
          int p1 = line.indexOf(',', 0);
          int p2 = line.indexOf(',', p1+1);
          int p3 = line.indexOf(',', p2+1);
          int p4 = line.indexOf(',', p3+1);
          String val = (p4 > 0)
                      ? line.substring(p3+1, p4)
                      : line.substring(p3+1);
          *counts[i] = val.toInt();
          Serial.printf("%s = %d\n", fields[i], *counts[i]);
        } else {
          Serial.printf("No data line for %s\n", fields[i]);
        }
      } else {
        Serial.printf("Failed fetching %s: HTTP %d\n", fields[i], code);
      }

      http.end();
    }

    // exactly one place to recompute and log total:
    totalCount = cowCount + goatCount + henCount;
    Serial.printf(
      "Restored counts → cows: %d, goats: %d, hens: %d, total: %d\n",
      cowCount, goatCount, henCount, totalCount
    );
  }  // ← this is the closing brace of fetchLastCounts()




  void updateOLED() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0,  0);
    display.printf("Cows:%d Goats:%d Hens:%d", cowCount, goatCount, henCount);
    display.setCursor(0, 12);
    display.printf("Total:%d", totalCount);
    display.setCursor(0, 24);
    display.printf("Last Conf: %.2f", lastConfidence);
    display.display();
  }

  void publishInfluxDB(int label, const String& direction) {
    if (WiFi.status() != WL_CONNECTED) return;

    // get current time in nanoseconds
    time_t now = time(nullptr);
    char ts[32];
    snprintf(ts, sizeof(ts), "%lld", (int64_t)now * 1000000000LL);

    // event names
    static const char* names[3] = { "cow", "goat", "hen" };

    // Build line protocol:
    // 1) single event point
    String lp = String("animal_tracker,animal=") +
                names[label] +
                ",direction=" + direction +
                " value=1 " +
                ts +
                "\n";

    // 2) summary point
    lp += String("animal_counts cowCount=")   + String(cowCount) +
          String(",goatCount=") + String(goatCount) +
          String(",henCount=")  + String(henCount)  +
          " " + ts;

    // send to InfluxDB
    HTTPClient http;
    String url = String(influx_host) +
                "/api/v2/write?org=" + influx_org +
                "&bucket=" + influx_bucket +
                "&precision=ns";
    http.begin(url);
    http.addHeader("Authorization", "Token " + String(influx_token));
    http.addHeader("Content-Type", "text/plain");
    
    int code = http.POST(lp);
    if (code > 0) {
      Serial.printf("Influx write OK: %d\n", code);
    } else {
      Serial.printf("Influx write failed: %s\n", http.errorToString(code).c_str());
    }
    http.end();
  }

  void classifyAndUpdate(const String& direction) {
    Serial.println("Starting classification...");

    // Fetch image from ESP32-CAM
    HTTPClient http;
    http.begin(cam_ip);
    if (http.GET() != HTTP_CODE_OK) {
      Serial.println("Failed to fetch image");
      http.end();
      return;
    }

    WiFiClient* stream = http.getStreamPtr();
    File tmp = SPIFFS.open("/tmp.jpg", FILE_WRITE);
    if (!tmp) {
      Serial.println("Failed to open /tmp.jpg for writing");
      http.end();
      return;
    }

    uint8_t buf[512];
    while (stream->available()) {
      size_t len = stream->readBytes(buf, sizeof(buf));
      tmp.write(buf, len);
    }
    tmp.close();
    http.end();

   // … after tmp.close() and http.end():
vTaskDelay(50 / portTICK_PERIOD_MS);

File chk = SPIFFS.open("/tmp.jpg", FILE_READ);
size_t fsize = chk.size();
chk.close();
Serial.printf("Saved JPEG size: %u bytes\n", fsize);
if (fsize < 1000) {
  Serial.println("File too small, aborting decode.");
  SPIFFS.remove("/tmp.jpg");
  return;
}

if (!JpegDec.decodeFsFile("/tmp.jpg")) {
  Serial.println("JPEG decoding failed");
  JpegDec.abort();
  SPIFFS.remove("/tmp.jpg");
  return;
}
// … continue with your resize, invoke, etc.


    Serial.printf("JPEG dimensions: %d x %d\n", JpegDec.width, JpegDec.height);
    Serial.println("Resizing and filling tensor...");

    for (int y = 0; y < 128; y++) {
      for (int x = 0; x < 128; x++) {
        int sx = map(x, 0, 127, 0, JpegDec.width - 1);
        int sy = map(y, 0, 127, 0, JpegDec.height - 1);
        uint16_t p = JpegDec.pImage[sy * JpegDec.width + sx];

        uint8_t r = ((p >> 11) & 0x1F) * 255 / 31;
        uint8_t g = ((p >> 5) & 0x3F) * 255 / 63;
        uint8_t b = (p & 0x1F) * 255 / 31;

        int idx = (y * 128 + x) * 3;
        input->data.uint8[idx + 0] = r;
        input->data.uint8[idx + 1] = g;
        input->data.uint8[idx + 2] = b;
      }
    }

    Serial.println("Invoking model...");
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Model invocation failed");
    //JpegDec.close();  // Remove this line
    JpegDec.abort();     // Optional, if you want to release JPEG buffer explicitly
    SPIFFS.remove("/tmp.jpg");
    return;
  }


    int best = 0;
    uint8_t max_val = output->data.uint8[0];
    for (int i = 1; i < output->dims->data[1]; ++i) {
      Serial.printf("Output[%d] = %d\n", i, output->data.uint8[i]);
      if (output->data.uint8[i] > max_val) {
        max_val = output->data.uint8[i];
        best = i;
      }
    }

    lastConfidence = max_val / 255.0f;
    Serial.printf("Best class: %d, confidence: %.2f\n", best, lastConfidence);

    if (lastConfidence < 0.7f) {
      Serial.println("Low confidence. Skipping update.");
      JpegDec.abort();
      SPIFFS.remove("/tmp.jpg");
      return;
    }

    // Update counts
    if (direction == "IN") {
      if      (best == 0) cowCount++;
      else if (best == 1) goatCount++;
      else                henCount++;
      totalCount++;
    } else {
      if      (best == 0 && cowCount > 0)   { cowCount--; totalCount--; }
      else if (best == 1 && goatCount > 0)  { goatCount--; totalCount--; }
      else if (henCount > 0)                { henCount--; totalCount--; }
    }

    updateOLED();
    publishInfluxDB(best, direction);

    // Proper cleanup
    JpegDec.abort();  // release image buffer and file
    SPIFFS.remove("/tmp.jpg");
    Serial.println("Classification complete\n");
  }
