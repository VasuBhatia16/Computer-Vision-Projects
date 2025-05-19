# 🧠 Computer Vision Projects

This repository contains real-time computer vision applications built using Python, OpenCV, and MediaPipe. Each project focuses on detecting and tracking different body landmarks using webcam input.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/VasuBhatia16/Computer-Vision-Projects.git
cd Computer-Vision-Projects
```

### 2. Create a Virtual Environment (Optional)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Most of the projects use the following libraries:

```bash
pip install opencv-python mediapipe
```

If needed, install others:

```bash
pip install numpy
pip install autopy
```

---

## 🚀 How to Run Each Project

### Face Detection

```bash
cd FaceDetectionProject
python FaceDetectionModule.py
```

### Face Mesh

```bash
cd FaceMeshProject
python FaceMeshModule.py
```

### Hand Tracking
#### Contains 3 projects
- Finger Counter
```bash
cd HandTrackingProject
python HandTrackingModule.py
```
- Gesture Volume Control
```bash
cd HandTrackingProject
python HandTrackingModule.py
```
- Virtual Hand Mouse
```bash
cd HandTrackingProject
python HandTrackingMouse.py
```
### Pose Estimation

```bash
cd PoseEstimationProject
python PoseEstimationModule.py
```

Make sure your webcam is connected and accessible. Each script opens a video feed and performs real-time detection.

---


## 📬 Contact

For questions or suggestions, reach out via:

- GitHub: [VasuBhatia16](https://github.com/VasuBhatia16)

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
