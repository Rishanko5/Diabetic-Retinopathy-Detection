🩺 Diabetic Retinopathy Detection using Deep Learning
📘 Overview
This project aims to detect Diabetic Retinopathy (DR) — a complication of diabetes that affects the eyes — using deep learning techniques.
The system analyzes retinal fundus images and classifies them into severity stages, helping doctors and healthcare professionals detect the disease early.

🎯 Objectives
• Automate the detection of Diabetic Retinopathy from retinal images.
• Achieve high accuracy using Convolutional Neural Networks (CNN).
• Support early diagnosis and reduce manual screening workload.

🧠 Project Workflow
1. Data Preprocessing
• Images are resized and normalized.
• Data augmentation techniques (rotation, flipping, zooming) improve generalization.
• Dataset is split into training, validation, and testing sets.
2. Model Building
• CNN-based deep learning model (Keras/TensorFlow) is used.
• Model trained on preprocessed dataset for multiple DR stages.
3. Model Evaluation
• Accuracy, Precision, Recall, and F1-score used for performance evaluation.
• Confusion matrix and classification report generated.
4. Prediction
• Trained model predicts DR severity from new retinal images.

🧾 Dataset
• Source: Kaggle – Diabetic Retinopathy Detection Dataset
• Input: Retinal fundus images
• Labels:
o 0 – No DR
o 1 – Mild
o 2 – Moderate
o 3 – Severe
o 4 – Proliferative DR

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/yourusername/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
2. Create and activate a virtual environment
python -m venv venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
venv\Scripts\activate      # (Windows)
# OR
source venv/bin/activate   # (Mac/Linux)
3. Install dependencies
pip install -r requirements.txt
4. Run the model
python main.py

🧩 Folder Structure
Diabetic-Retinopathy-Detection/
│
├── data/                     # Dataset folder (images)
├── models/                   # Saved trained models (.h5)
├── notebooks/                # Jupyter notebooks for EDA/experiments
├── src/                      # Main Python scripts
│   ├── preprocess.py         # Image preprocessing
│   ├── train_model.py        # Model training
│   ├── evaluate.py           # Model evaluation
│   └── predict.py            # Prediction on new images
│
├── requirements.txt          # Project dependencies
├── main.py                   # Entry point script
├── README.md                 # Project documentation
└── results/                  # Evaluation metrics, confusion matrix, plots

📊 Model Performance
MetricScoreAccuracy93%Precision92%Recall91%F1 Score91%(Values may vary depending on dataset and training parameters.)

🧩 Technologies Used
• Python 3.10+
• TensorFlow / Keras
• OpenCV
• NumPy, Pandas
• Matplotlib, Seaborn
• Scikit-learn

🚀 Future Enhancements
• Implement transfer learning with EfficientNet or ResNet.
• Deploy as a web app using Streamlit or Flask.
• Integrate Grad-CAM for explainable AI visualization.
• Expand to multi-disease detection (Glaucoma, Cataract, etc.)

🧑‍💻 Contributors
• Rishank Kumbhare — Machine Learning Developer


🩷 Acknowledgements
• Kaggle for dataset access.
• TensorFlow and Keras open-source communities.
• Medical research teams contributing to DR detection datasets.

📜 License
This project is licensed under the MIT License — you are free to use, modify, and distribute this work with proper attribution.

