#define IR_A 6
#define IR_B 5

enum DirectionState {
  IDLE,
  WAITING_FOR_B, // A triggered
  WAITING_FOR_A  // B triggered
};

DirectionState state = IDLE;

unsigned long triggerTime = 0;
const unsigned long timeout = 1000; // max time allowed between triggers (ms)

void setup() {
  Serial.begin(115200);
  pinMode(IR_A, INPUT_PULLUP);
  pinMode(IR_B, INPUT_PULLUP);
  Serial.println("IR Direction Detection System Ready");
}

void loop() {
  bool a = digitalRead(IR_A);
  bool b = digitalRead(IR_B);
  unsigned long now = millis();

  // Show live sensor values
  Serial.print("IR_A: "); Serial.print(a);
  Serial.print(" | IR_B: "); Serial.print(b);
  Serial.print(" | State: "); Serial.println(state);

  switch (state) {
    case IDLE:
      if (a == LOW) {
        state = WAITING_FOR_B;
        triggerTime = now;
        Serial.println("A triggered → Waiting for B");
      } else if (b == LOW) {
        state = WAITING_FOR_A;
        triggerTime = now;
        Serial.println("B triggered → Waiting for A");
      }
      break;

    case WAITING_FOR_B:
      if (b == LOW) {
        Serial.println("👉 Direction: IN (A → B)");
        // Trigger camera for IN
        state = IDLE;
      } else if (now - triggerTime > timeout) {
        Serial.println("⏱ Timeout - reset");
        state = IDLE;
      }
      break;

    case WAITING_FOR_A:
      if (a == LOW) {
        Serial.println("👈 Direction: OUT (B → A)");
        // Trigger camera for OUT
        state = IDLE;
      } else if (now - triggerTime > timeout) {
        Serial.println("⏱ Timeout - reset");
        state = IDLE;
      }
      break;
  }

  delay(20); // sampling interval
}
