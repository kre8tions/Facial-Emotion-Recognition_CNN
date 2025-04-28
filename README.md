Facial Emotion Recognition with CNN
This project implements a Convolutional Neural Network (CNN) to detect and classify human emotions from facial images using the FER2013 dataset. The model identifies seven emotions: angry, disgust, fear, happy, sad, surprise, and neutral. Built with Python, TensorFlow/Keras, and OpenCV, the project includes data preprocessing, model training, evaluation, and optional real-time emotion detection via webcam.
Table of Contents
Project Overview (#project-overview)

Dataset (#dataset)

Installation (#installation)

Usage (#usage)

Project Structure (#project-structure)

Results (#results)

Contributing (#contributing)

License (#license)

Project Overview
The goal is to train a CNN to recognize emotions in facial images, a key computer vision task with applications in human-computer interaction, accessibility, and sentiment analysis. The model is trained on the FER2013 dataset, which contains 35,887 grayscale images labeled with seven emotions. Key features include:
Data preprocessing and augmentation for robust training.

A CNN architecture with convolutional, pooling, and dense layers.

Model evaluation on a test set with accuracy and confusion matrix.

Optional face detection and real-time emotion recognition using OpenCV.

This project is inspired by the MIT Intro to Deep Learning course, demonstrating core concepts like feature extraction and classification.
Dataset
FER2013 Dataset:
Description: Contains 35,887 grayscale images (48x48 pixels) labeled with seven emotions: angry, disgust, fear, happy, sad, surprise, neutral.

Source: Kaggle FER2013 Dataset

Structure: Organized into train and test folders, with subfolders for each emotion (e.g., happy, sad).

Download: Requires a Kaggle account. Place the extracted fer2013 folder in the project root.

Installation
Prerequisites
Python 3.8–3.10

Git

Kaggle account (for dataset download)

Optional: Webcam for real-time testing

Steps
Clone the Repository:
bash

git clone https://github.com/kre8tions/Facial-Emotion-Recognition_CNN.git
cd Facial-Emotion-Recognition_CNN

Set Up a Virtual Environment:
bash

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

Install Dependencies:
bash

pip install tensorflow numpy pandas opencv-python matplotlib

Download the FER2013 Dataset:
Download from Kaggle and extract the fer2013 folder to the project root.

Alternatively, use the Kaggle API:
bash

pip install kaggle
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip

Usage
Training the Model
Ensure the fer2013 dataset is in the project root.

Run the training script:
bash

python train.py

Loads the dataset, preprocesses images, trains the CNN, and saves the model as model.h5.

Training may take 10–60 minutes depending on hardware (GPU recommended).

Testing the Model
Test on a single image:
bash

python predict.py --image path/to/image.jpg

Real-time webcam testing (requires OpenCV and a webcam):
bash

python webcam_predict.py

Visualizing Results
After training, check the generated confusion matrix and classification report in the console or saved plots.

Use Jupyter notebooks in the notebooks/ folder for exploratory data analysis (EDA).

Project Structure

Facial-Emotion-Recognition_CNN/
├── fer2013/                  # Dataset folder (train/, test/)
├── venv/                     # Virtual environment
├── .gitignore                # Git ignore file
├── README.md                 # Project documentation
├── train.py                  # Script for training the CNN
├── predict.py                # Script for predicting emotions on images
├── webcam_predict.py         # Script for real-time webcam prediction
├── model.h5                  # Trained model file
└── notebooks/                # Jupyter notebooks for EDA and visualization

Results
Model Performance:
Test accuracy: Approximately 60–65% (varies with hyperparameters and training epochs).

Confusion matrix and classification report generated post-training.

Sample Output:
Predicted emotions on test images with confidence scores (e.g., "Happy: 0.85").

Real-time webcam demo displays emotion labels overlaid on detected faces.

Visualizations:
Training/validation loss and accuracy plots.

Confusion matrix showing model performance across emotions.

Contributing
Contributions are welcome! To contribute:
Fork the repository.

Create a feature branch:
bash

git checkout -b feature/your-feature

Commit changes:
bash

git commit -m "Add your feature"

Push to the branch:
bash

git push origin feature/your-feature

Open a pull request.

Please adhere to PEP 8 coding standards and include tests for new features.
License
This project is licensed under the MIT License. See the LICENSE file for details.

