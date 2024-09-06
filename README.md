# Brain Tumor Detection using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to detect brain tumors using image data. The dataset consists of brain scan images, categorized into two classes: **'yes' (tumor present)** and **'no' (no tumor)**. The model processes the images, classifies them, and predicts whether or not a tumor is present.

## Dataset
The dataset consists of brain MRI images labeled with two categories:
- **Yes**: Tumor present.
- **No**: No tumor.

Each image is resized to **128x128 pixels** and converted into an appropriate format for use in a CNN model. The dataset is split into training and test sets, with data augmentation applied to improve the model's generalization.

## Requirements
The following libraries and frameworks are used in this project:
- **Python 3.x**
- **TensorFlow/Keras**: Used for building the CNN.
- **OpenCV**: For image processing and resizing.
- **NumPy**: For handling arrays.
- **Pandas**: For data manipulation.
- **Seaborn & Matplotlib**: For visualizing results.
- **scikit-learn**: For data splitting, metrics, and evaluation.
  
Install the necessary libraries using the following:
```bash
pip install numpy pandas opencv-python tensorflow keras seaborn matplotlib scikit-learn
```

## Project Workflow

### 1. Importing Libraries
The necessary libraries such as Keras, OpenCV, NumPy, Matplotlib, and others are imported.

### 2. Loading the Dataset
- The dataset is loaded from a local directory. It contains two folders: 'yes' and 'no', each containing images.
- The images are read using OpenCV and resized to 128x128 pixels.
- The labels for the images are assigned: **0 for 'yes' (tumor)** and **1 for 'no' (no tumor)**.

### 3. Data Visualization
- A subset of images is randomly selected and displayed along with their labels (tumor/no tumor) using Matplotlib for visualization.
  
### 4. Splitting the Data
- The data is split into **training (70%)** and **testing (30%)** sets using `train_test_split` from scikit-learn.

### 5. Data Distribution Visualization
- Histograms are generated to show the distribution of the classes (tumor/no tumor) in both training and testing datasets.

### 6. Data Preprocessing
- Images are converted to grayscale and normalized to have pixel values between 0 and 1.
- The images are reshaped to have a shape of (128, 128, 1) to fit the CNN model input requirements.

### 7. Data Augmentation
- **ImageDataGenerator** from Keras is used to augment the training data, applying random transformations such as shifts, zooms, and rotations to improve the robustness of the model.

### 8. Building the CNN Model
- A sequential CNN model is constructed using Keras:
  - Several **Conv2D** layers with ReLU activation are applied for feature extraction.
  - **MaxPooling2D** layers reduce the spatial dimensions.
  - **Dropout** layers help prevent overfitting.
  - **Dense** layers are added at the end for classification.
- The model is compiled using **Adam** optimizer and **categorical cross-entropy** loss.
  
### 9. Training the Model
- The model is trained for **30 epochs** with a batch size of **40**. The training and validation loss/accuracy are recorded for analysis.

### 10. Results & Visualization
- The training and validation loss/accuracy are plotted using Matplotlib.
- The trained model is evaluated on the test set, and the accuracy and loss are printed.
  
### 11. Confusion Matrix
- A confusion matrix is generated to visualize the performance of the model in classifying images. This is done using the `confusion_matrix` function from scikit-learn and visualized using Seaborn.

## Performance Metrics
- **Accuracy**: The model's accuracy on the test set is displayed.
- **Confusion Matrix**: A confusion matrix is generated to evaluate the classification performance.

## How to Run
1. Clone the repository and place your dataset in the correct directory structure:
   ```
   brain_tumor_dataset/
      ├── yes/
      └── no/
   ```
2. Ensure all the necessary libraries are installed.
3. Run the script to load and preprocess the dataset, and train the model.

## Conclusion
This project demonstrates the use of CNNs for image classification tasks, specifically for detecting brain tumors in MRI images. Data augmentation, preprocessing, and a well-constructed CNN model contribute to its classification performance.

