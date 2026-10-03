#include <TinyGPSPlus.h>

// Create the TinyGPS++ parsing object
TinyGPSPlus gps;

// Define the hardware serial port for the GPS
#define gpsSerial Serial2

// Timer variable to handle non-blocking printing
unsigned long lastPrintTime = 0;

void setup() {
  // 1. Initialize communication with your PC Serial Monitor (Set to 115200)
  Serial.begin(115200);
  
  // 2. Initialize communication with your GPS module on your ACTUAL pins:
  // Your blue wire (GPS TX) goes to ESP32 RX2 (Pin 16)
  // Your purple wire (GPS RX) goes to ESP32 D4 (Pin 4)
  gpsSerial.begin(9600, SERIAL_8N1, 25, 26);
  
  Serial.println("System Initialized. Waiting for GPS data stream...");
}

void loop() {
  // 1. Constantly feed data to TinyGPS++ without any delay interruptions.
  // This keeps the internal buffer clear so data doesn't get corrupted.
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  // 2. Safely print the layout table every 2 seconds without using blocking delay()
  if (millis() - lastPrintTime >= 2000) {
    lastPrintTime = millis();
    displayLocationInfo();
  }

  // 3. Simple wiring check that doesn't freeze or lock up the code
  if (millis() > 6000 && gps.charsProcessed() < 10) {
    Serial.println(F("WARNING: No raw GPS bytes received yet. Check RX/TX connections."));
  }
}

void displayLocationInfo() {
  Serial.println(F("-------------------------------------"));
  Serial.println(" Location Info:");

  // Latitude
  Serial.print("Latitude:   ");
  if (gps.location.isValid()) {
    Serial.print(gps.location.lat(), 6);
    Serial.print(" ");
    Serial.println(gps.location.rawLat().negative ? "S" : "N");
  } else {
    Serial.println("Searching for satellites...");
  }

  // Longitude
  Serial.print("Longitude:  ");
  if (gps.location.isValid()) {
    Serial.print(gps.location.lng(), 6);
    Serial.print(" ");
    Serial.println(gps.location.rawLng().negative ? "W" : "E");
  } else {
    Serial.println("Searching for satellites...");
  }

  // Fix Status and Satellites
  Serial.print("Fix Status: ");
  Serial.println(gps.location.isValid() ? "VALID FIX" : "NO FIX (Go near a window!)");

  Serial.print("Satellites: ");
  Serial.println(gps.satellites.value());

  // Altitude
  Serial.print("Altitude:   ");
  Serial.print(gps.altitude.meters());
  Serial.println(" m");

  // Speed
  Serial.print("Speed:      ");
  Serial.print(gps.speed.kmph());
  Serial.println(" km/h");

  // Date & Time
  Serial.print("Date:       ");
  if (gps.date.isValid()) {
    Serial.printf("%02d/%02d/%04d\n", gps.date.day(), gps.date.month(), gps.date.year());
  } else {
    Serial.println("Waiting...");
  }

  Serial.print("Time (UTC): ");
  if (gps.time.isValid()) {
    Serial.printf("%02d:%02d:%02d\n", gps.time.hour(), gps.time.minute(), gps.time.second());
  } else {
    Serial.println("Waiting...");
  }

  Serial.println(F("-------------------------------------"));
}