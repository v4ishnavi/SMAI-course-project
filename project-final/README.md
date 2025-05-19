# SMAI PROJECT SPRING 2025 - 2022102070

## Region ID

This task focused on classifying input images into one of several predefined Region IDs. I explored multiple architectures to optimize classification accuracy. I started with a pretrained Vision Transformer (ViT), achieving ~91% accuracy. Switching to a YOLOv8m-classification model boosted performance to 95.1%. I also trained an EfficientNet-B0 (~92.5%) and finally a YOLOv11m-classification model, which achieved the best performance with 95.4% accuracy.

The YOLOv11m-classification model is a convolutional neural network designed for both detection and classification tasks. It integrates fast feature extraction with efficient training using Ultralytics’ framework, making it well-suited for deployment.

### Preprocessing Steps:
- Structured the dataset to match YOLO’s requirements: images are stored in `train/` and `val/` folders, each with subfolders for every class.
- Created a `.yaml` config file specifying class names, training paths, and number of classes.
- Converted CSV label files into YOLO-compatible folder structures.
- Used 224×224 resolution and leveraged YOLO's built-in augmentations.
- Training was done with early stopping and performance monitored via validation accuracy.

---

## Angle ID

This was a regression task to estimate the **angle of orientation** from images. To handle the cyclical nature of angles, I encoded the angle in two components: **sine** and **cosine** of the angle (in radians). This encoding prevents discontinuities around 0° and 360°.

### Pipeline Overview:
- Cleaned the dataset by removing invalid or outlier angles (e.g., angles > 360°).
- Converted angle in degrees → radians → sin and cos components (`angle_sin`, `angle_cos`).
- Trained an EfficientNet-B0 model to predict these two values using MSELoss.
- During inference, the predicted angle was reconstructed using `arctan2(sin, cos)` to obtain the actual angle.
- Image preprocessing was the same as in other modules (resizing, normalization).
- This method ensured smooth learning and better generalization across all orientations.

---

## LatLong

This module involved regressing **latitude and longitude** values from images. I used deep CNNs to predict location coordinates with robust preprocessing and outlier handling to improve spatial consistency.

### Preprocessing Pipeline:
1. **Absolute Pathing**: All image paths were resolved for stable access.
2. **DBSCAN Clustering**: For each Region ID, I ran DBSCAN on the lat-long coordinates to detect and discard spatial outliers that didn’t fit within the main cluster.
3. **Quartile Filtering**: Further refined data by removing values outside the 25–75 percentile (IQR) range in both latitude and longitude dimensions.
4. **Normalization**: Normalized latitude and longitude using training set mean and standard deviation for numerical stability during training.
5. **Image Resizing**: All input images were resized to 224×224 resolution.
6. **Dataset Balancing**: Ensured each region had a balanced number of samples after cleaning to prevent model bias.

### Modeling and Evaluation:
- Trained EfficientNet-B0 using MSE loss to predict [lat, lon].
- Used `ReduceLROnPlateau` scheduler and early stopping for optimal convergence.
- Predictions were de-normalized during evaluation and plotted on scatter plots to check real-world consistency.

This pipeline effectively combined geospatial cleaning (DBSCAN, quartile filtering) with strong CNN regressors to build a robust LatLong predictor.

---
