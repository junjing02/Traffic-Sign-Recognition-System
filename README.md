# Traffic Sign Detection & Recognition System

**Course:** Mini Project

## Project Overview  
A real-time C++/OpenCV application that detects, segments, and classifies road traffic signs to support autonomous vehicles and ADAS. The pipeline consists of:  
1. **Color Segmentation:** HSV thresholding to isolate red, blue, and yellow sign regions.  
2. **Shape Filtering:** Douglas–Peucker contour approximation to detect triangles and circles.  
3. **Region Extraction & Preprocessing:** Crop, resize to 64×64 px, convert to HSV and gray scale.  
4. **Feature Extraction:** HOG descriptor plus one-hot color encoding.  
5. **Classification:** Train and test both SVM and Random Forest models; evaluate via confusion matrix and accuracy.

## Technical Details  
- **Language & Standard:** C++ 
- **Dependencies:**  
  - OpenCV  
- **Key Source Files:**  
  - `Source.cpp` – main application and pipeline logic  
  - `supp.cpp` / `Supp.h` – utility functions for window partitioning, normalization, Gaussian kernel, etc.   
- **Algorithms & Parameters:**  
  - **HSV Ranges:** red (0–10 & 170–220 H), yellow (10–40 H), blue (90–130 H)  
  - **ApproxPolyDP Tolerances:** 6% of contour length for triangles, 1% for circles  
  - **HOG Descriptor:** winSize 64×64, blockSize 16×16, cellSize 8×8, nbins 9  
  - **SVM:** Linear kernel, C = 1, γ = 0.5, C_SVC  
  - **Random Forest:** 50 trees, maxDepth 10, minSampleCount 2, activeVarCount 4

## Build & Run

**Prepare Data & Models**

   * Place your traffic sign images under `Inputs/train/` and `Inputs/signs/`.
   * If you have pre-trained models, put `traffic_sign_svm.xml` and `traffic_sign_rf.xml` in `models/`. Otherwise, the program will train new ones.

**Run the Application**

   Choose from the menu:

   1. Train SVM
   2. Test SVM
   3. Train Random Forest
   4. Test Random Forest
   5. Predict Single Image (SVM)

   For option 5, enter the filename (e.g. `005.png`) located in `Inputs/input/`.

**Results**

   * Confusion matrix and low-precision/recall classes are printed to console.
   * Accuracy and number of correctly segmented signs are displayed.
   * OpenCV windows show original vs. segmented images and HOG feature plots.

