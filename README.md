# Detecting COVID-19 with Chest X-Ray using PyTorch

Train a model on a [COVID-19 Radiography dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/download?datasetVersionNumber=5) from kaggle, comprising nearly 3000 Chest X-Ray scans categorized into Normal, Viral Pneumonia, and COVID-19 classes. A practical implementation of Convolutional Neural Networks (CNNs) and optimization techniques.

## Learning Objectives
- Create custom Dataset and DataLoader in PyTorch
- Train a ResNet model for Image Classification

## Skills Practiced
- Machine Learning
- Deep Learning
- Statistical Classification
- PyTorch
- Medical Imaging

## Model Performance

After training the model for 30 epochs, the following performance metrics were achieved on the test set:

- Test Accuracy: 87.25%

### Classification Report for Test Set

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Normal   | 0.97      | 0.85   | 0.91     | 1019    |
| Viral    | 0.70      | 0.96   | 0.81     | 134     |
| COVID-19 | 0.74      | 0.89   | 0.81     | 361     |

- Overall Accuracy: 87.25%
- Weighted F1-Score: 0.88