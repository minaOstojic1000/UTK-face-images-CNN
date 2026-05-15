# UTKFace Image Classification with Transfer Learning

This is an academic computer vision project focused on image classification using transfer learning. The project was developed as part of a university assignment, with the goal of learning how an existing convolutional neural network can be adapted to a specific dataset and evaluated using standard machine learning metrics.

The model was trained on the UTKFace dataset, which contains face images annotated with age, gender, and ethnicity labels. For this project, the classification task was based on five dataset categories: `white`, `black`, `asian`, `indian`, and `other`.

> Note: This project was created for educational purposes only. Since it involves sensitive human-related image categories, the results should not be used for real-world decision-making or identity-related applications.

## Project Overview

The main goal of this project was to apply transfer learning to an image classification problem. Instead of building a convolutional neural network completely from scratch, we used a pre-trained MobileNetV2 model and adapted it to our classification task.

The project included:

- dataset preprocessing
- train/validation/test split
- image normalization
- data augmentation
- class weighting for imbalanced classes
- transfer learning with MobileNetV2
- model training and evaluation
- analysis of accuracy, precision, recall, F1-score, and confusion matrices

## Dataset

The project uses the UTKFace dataset:

https://www.kaggle.com/datasets/jangedoo/utkface-new

The dataset contains 23,705 face images. In this project, images were resized to 96×96 pixels and classified into five categories:

- `white`
- `black`
- `asian`
- `indian`
- `other`

The dataset is imbalanced, with the `white` class being the most represented and the `other` class being the least represented. To reduce the effect of class imbalance, class weighting was applied during training.

## Model

The model is based on MobileNetV2, a convolutional neural network architecture commonly used for efficient image classification.

The MobileNetV2 base was used as a feature extractor, while the classification part was trained for this specific task. The project used a one-phase transfer learning approach, where the base model weights were frozen and only the custom classification layers were trained.

Techniques used to improve training and reduce overfitting:

- data augmentation
- dropout
- batch normalization
- early stopping
- class weighting

## Preprocessing and Augmentation

The preprocessing pipeline included:

- resizing images to 96×96 pixels
- normalizing pixel values from `[0, 255]` to `[0, 1]`
- splitting the dataset into training, validation, and test sets

Data augmentation was applied only to the training set and included:

- random horizontal flipping
- random rotation
- random zoom
- random translation

## Results

The model achieved moderate performance on the test set. The final evaluation results were approximately:

| Metric | Test Set |
|--------|----------|
| Accuracy | 60.13% |
| Precision | 60.55% |
| Recall | 60.13% |
| F1-score | 60.14% |

The confusion matrix showed that some classes were easier to classify than others. The `white` class achieved the highest accuracy, while the `other` class was the most difficult to classify, which was expected due to its lower representation and more heterogeneous nature.

## Responsible AI Considerations

This project involves sensitive human-related image categories. Because of that, it is important to emphasize several limitations:

- The dataset is imbalanced.
- The labels may contain bias or ambiguity.
- Visual classification of ethnicity-related categories is a sensitive and subjective task.
- The model performance is limited and should not be interpreted as reliable for real-world use.
- The project should be treated only as an academic exercise in transfer learning and model evaluation.

Working on this project helped us understand not only the technical side of AI, but also the importance of responsible AI development, dataset bias, and careful interpretation of model results.

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- scikit-learn

## Authors

This project was developed as a university project by two students.

## How to Run

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the UTKFace dataset from Kaggle:

https://www.kaggle.com/datasets/jangedoo/utkface-new

4. Place the dataset in the expected local folder, for example:

```text
data/UTKFace/
```

5. Run the training script or notebook:

```bash
python main.py
```

or open the notebook:

```bash
jupyter notebook
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluation.py
├── notebooks/
│   └── utkface_classification.ipynb
├── results/
│   ├── accuracy_loss_plot.png
│   ├── confusion_matrix_test.png
│   └── confusion_matrix_train.png
└── report/
    └── project_report.pdf
```

The dataset itself is not included in the repository because of its size and licensing considerations. It should be downloaded separately from Kaggle.
