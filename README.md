# 🧘‍♀️ OMline Yoga Pose Corrector

A real-time AI-powered yoga pose correction system built using computer vision and machine learning. This application guides users through yoga asanas with visual and voice feedback, making home practice both effective and safe.

---

## 🔍 Overview

The OMline Yoga Pose Corrector is designed to:
- 📷 Capture real-time human pose using MediaPipe.
- 📐 Extract joint angle features from images.
- 🤖 Train a Random Forest Classifier to classify yoga poses with high accuracy (99.1%).
- 🗣️ Provide real-time feedback with pose correction instructions via both visual cues and voice (text-to-speech).
- 💻 Deliver an intuitive GUI-based user experience for end users.

---

## 🧱 Project Structure

| File / Folder | Description |
|---------------|-------------|
| `YogaPoseDataCollector.ipynb` | Captures pose images, extracts joint angles, and generates `yoga_pose_angles_dataset.csv` and `test.csv`. Also trains a Random Forest Classifier. |
| `pose_classifier.pkl`, `label_encoder.pkl` | Saved model and label encoder used in inference. |
| `yoga_pose_corrector_GUI.py` | Main GUI script to run the application. Connects to the webcam and backend. |
| `yoga_pose_detector.py` | Handles pose detection and feedback logic (visual + voice) using MediaPipe and TTS. |
| `DATASET/` | Folder containing captured yoga pose images. |
| `Omline.png`, `posecorrector_UI.png` | Project logo and GUI visuals. |


---

## 📊 Model Performance

- **Model Used**: Random Forest Classifier  
- **Accuracy**: `99.14%`  
- **Features**: Joint angles calculated using MediaPipe pose landmarks  
- **Dataset**: Custom dataset of yoga pose images  
- **Exported Files**: `yoga_pose_angles_dataset.csv`, `test.csv`

---

## 🧪 Requirements

Install Python dependencies using pip:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run the Application

⚠️ Ensure your webcam is connected and working.
- 1. Clone the Repo
```bash 
git clone https://github.com/ananya-03/OMline_Yoga_Pose_Corrector.git
cd OMline_Yoga_Pose_Corrector
```
-  Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
- Install Dependencies
```bash
pip install -r requirements.txt
```
- Run the main GUI:
```bash
python yoga_pose_corrector_GUI.py
```
The GUI will launch and connect to the webcam to start detecting poses and providing feedback.

## 🧘‍♂️ Yoga Day Launch Special 🎉

### This project was built to commemorate International Yoga Day, showcasing how AI can contribute to health, wellness, and inclusive home workouts — especially for the elderly or beginners.

## 📂 To-Do / Future Work

- Add more pose categories (e.g., dynamic sequences)
- Mobile-friendly deployment (e.g., using TFLite or ONNX)
- Feedback personalization based on user performance
- Support multiple languages for TTS

## 🧑‍💻 Author
### Ananya Singh

## 📜 License

This project is open-source and available under the MIT License.