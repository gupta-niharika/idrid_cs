## Disease Grading Classification Workflow

1. 📁 **Set up paths**  
    Defined all directory and file paths for images, masks, and CSVs.

2. 🗂️ **Copy images for classification**  
    Wrote a helper function to copy training and testing images into dedicated folders for classification.

3. 🏷️ **Create custom Dataset class**  
    Implemented `IDRiDClassificationDataset` to load images and labels from CSVs.

4. 🖼️ **Define image transforms**  
    Used torchvision transforms for data augmentation (train) and normalization (train/test).

5. 🗺️ **Map labels and create datasets**  
    Created `train_dataset` and `test_dataset` using the custom dataset class and transforms.

6. 📦 **Create DataLoaders**  
    Wrapped datasets in PyTorch DataLoaders for batching and shuffling.

7. 🏗️ **Build the model**  
    Defined a function to build a ResNet18 classifier with a custom output layer for 5 classes.

8. ⚖️ **Compute class weights**  
    Used `sklearn` to calculate class weights for handling class imbalance.

9. 🔁 **Training loop**  
    Implemented a training loop with early stopping, label smoothing, and class weights.

10. 💾 **Save the best model**  
     Saved the trained model weights to disk.

11. 🧪 **Evaluate on test set**  
     Evaluated the model on the test set and computed accuracy.

12. 📊 **Plot metrics**  
     Plotted training/validation loss and accuracy curves.

13. 🧮 **Confusion matrix**  
     Plotted a confusion matrix to visualize prediction performance across classes.

14. 📝 **Classification report**  
     Printed a detailed classification report (precision, recall, F1-score) for each class.

15. 📄 **Save predictions**  
     Saved true and predicted labels for the test set to a CSV file.

16. 👁️ **Visualize predictions**  
     Displayed sample test images with predicted and true labels for qualitative inspection.

17. 🗒️ **Log results**  
     Logged model performance metrics and notes to a CSV file for future reference.
----

## 🧩 Steps to Build the Multi-Task Dynamic Routing Model

1. **📁 Project Setup & Paths**
  - Define all dataset and directory paths for segmentation images, masks, grading images, and CSV labels.

2. **🧪 Data Transformations**
   - Use `albumentations` for image augmentation and normalization for both training and testing.

3. **📦 Multi-Task Dataset**
    - Implement a custom `IDRiDMultiTaskDataset` class:
    - Loads fundus images, segmentation masks (multi-channel), and disease grading labels.
    - Handles mask stacking for multiple lesions and applies transformations.

4. **🔍 Data Exploration**
   - Visualize images and their corresponding lesion masks (both individual and combined overlays).

5. **🏗️ Model Architecture**
   - Define `MultiTaskResNetWithRouting`:
   - Shared ResNet18 encoder.
   - Routing layer (soft task gate) to dynamically weight classification and segmentation tasks.
   - Separate expert heads for classification (disease grading) and segmentation (multi-channel lesion masks).

6. **⚖️ Loss Functions**
   - Implement `MultiChannelDiceLoss` for segmentation.
   - Use `CrossEntropyLoss` (with class weights) for classification.

7. **💻 Device Setup**
   - Specify computation device (CPU/GPU).

8. **📊 DataLoader Preparation**
   - Split dataset into training and validation sets.
   - Create PyTorch DataLoaders for efficient batching.

9. **🔁 Training Loop**
   - Train with early stopping:
   - Jointly optimize classification and segmentation losses, weighted by α and β.
   - Track and display training loss and accuracy.

10. **🔍 Hyperparameter Search**
    - Run experiments over different α and β values.
    - Compute class weights to handle label imbalance.
    - Store and display experiment results in a live table.

11. **💾 Model Saving**
    - Save the best-performing model checkpoint.

12. **📈 Results Visualization**
    - Plot training loss and accuracy curves for each experiment.
    - Visualize the effect of β on best accuracy.

13. **📋 Summary Table**
    - Highlight and sort experiment results for easy comparison and Save summary as CSV.

14. **🧪 Testing & Evaluation**
    - Load the best model.
    - Prepare test dataset and DataLoader.
    - Evaluate classification and segmentation performance (classification report, average Dice score).

15. **🖼️ Qualitative Visualization**
    - Overlay predicted and ground truth lesion masks on fundus images for visual inspection.

16. **🧮 Confusion Matrix**
    - Plot confusion matrix for classification results.

17. **⚔️ Performance Comparison**
    - Compare single-task and multi-task model performance using classification reports.

---
