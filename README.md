# Traffic Sign Detection & Recognition System

**Course:** Mini Project

## Project Overview
A C++/OpenCV application that detects, segments, and classifies road traffic signs. It supports both training and testing modes using two classifiers (SVM and Random Forest), and provides real-time visualization of segmented signs, HOG feature plots, and classification results.

## Technical Details
- **Language & Standard:** C++ 
- **Dependencies:** OpenCV
- **Core Modules:**  
  - **`Source.cpp`** – main pipeline:  
    - Color segmentation (red/blue/yellow) → shape filtering → crop & resize to 64×64  
    - One-hot color encoding + HOG feature extraction → feature matrix conversion  
    - Train & save SVM (`traffic_sign_svm.xml`) or Random Forest (`traffic_sign_rf.xml`) 
    - Load models to predict on test or input images, compute/print confusion matrix and accuracy  
    - Visualization: original vs segmented vs HOG-feature plots  
  - **`supp.cpp` / `Supp.h`** – utility functions:  
    - Window partitioning & legends for side-by-side display  
    - Grayscale & color normalization helpers  
    - Gaussian kernel generation   