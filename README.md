# Real-Time Driver Drowsiness Detection and Emergency Safe Parking

## Overview
The **Real-Time Driver Drowsiness Detection and Emergency Safe Parking System** is an embedded system designed to detect driver drowsiness in real time and safely park the vehicle in a secure location. The system integrates **Embedded Systems, CNN, Dlib, RF Communication, and Infrared Sensors** to ensure enhanced safety and reduced accidents.

## Features

### 1. Real-Time Drowsiness Detection
- Utilizes **CNN with Dlib** for **facial analysis** (eye closure and yawning detection).
- Edge AI inference for real-time processing on low-power embedded systems.

### 2. Emergency Safe Parking System
- Automatically initiates a **safe parking maneuver** if drowsiness is detected.
- Uses **RF Communication and Infrared Sensors** to monitor the surroundings and detect obstacles.

### 3. Enhanced Safety with Infrared Sensors
- **Integrated infrared sensors** with Arduino for real-time **object detection**.
- Achieved a **40% reduction in false alarms** during testing phases.

## Tech Stack
- **Embedded Systems:** Arduino, Raspberry Pi
- **Machine Learning:** CNN, Dlib
- **Communication Protocols:** RF Communication
- **Sensors:** Infrared Sensors

## Installation & Setup

### Prerequisites:
- **Arduino IDE & Required Libraries**
- **Python 3.8+**
- **OpenCV, Dlib, TensorFlow**

### Steps to Run:
#### Clone the Repository:
```sh
git clone https://github.com/Moni282003/Real-Time-Driver-Drowsiness-Detection.git
cd Real-Time-Driver-Drowsiness-Detection
```

#### AI Model Setup:
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
python app.py
```

---
**Monish M - Developer**

*Note: If you face any issues, feel free to open an issue on [GitHub](https://github.com/Moni282003/Driver_Drowsiness_Detection/issues).*

