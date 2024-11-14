# self-supervised-learning



Improving Time Series Classification Accuracy using Self-Supervised Learning

Project Overview

This project aims to enhance the classification accuracy of time series data, specifically for gesture recognition, using self-supervised learning (SSL). By pre-training a model with SSL on the Human Activity Recognition (HAR) dataset, we extract feature representations that improve classification performance on the UWaveGestureLibrary dataset.

Dataset

	1.	HAR Dataset: Used for self-supervised pre-training.
	2.	UWaveGestureLibrary Dataset: Used for training and evaluating the gesture classification model.

Dataset Instructions

	•	HAR Dataset and UWaveGestureLibrary Dataset: These files are located in the data/ directory within subdirectories HAR and Gesture, respectively. If they are not included due to size, download them from the following links:
	•	HAR Dataset: Available here.
	•	UWaveGestureLibrary Dataset: Available here.
	•	After downloading, place them in the data/ directory, structured as follows:

data/
    HAR/
        train.pt
        val.pt
        test.pt
    Gesture/
        train.pt
        val.pt
        test.pt



Project Structure

- data/
  - HAR/                # Folder containing HAR dataset (train.pt, val.pt, test.pt)
  - Gesture/            # Folder containing UWaveGestureLibrary dataset (train.pt, val.pt, test.pt)
- TimeSeriesClassification_SSL.ipynb   # Jupyter Notebook with all project code
- README.md                             # Project documentation
- requirements.txt                      # List of dependencies

Installation

To set up the project, clone the repository and install the required dependencies:

git clone [repository-url]
cd [repository-directory]
pip install -r requirements.txt

Usage

Open the Jupyter Notebook TimeSeriesClassification_SSL.ipynb to run the project. It contains all the steps from data preprocessing to model training and evaluation.
	1.	Data Preprocessing: Loads and scales the HAR and UWaveGestureLibrary datasets.
	2.	Self-Supervised Learning: Trains a Siamese Network using contrastive learning on the HAR dataset to learn feature representations.
	3.	Classification Model Training: Fine-tunes the classifier using the extracted embeddings on the UWaveGestureLibrary dataset.
	4.	Evaluation: Evaluates model performance on the test set and visualizes results.

Running the Notebook

Run all cells in the notebook sequentially to reproduce the workflow. The notebook is organized with explanations in Markdown cells, guiding you through each step.

Model Overview

	1.	Siamese Network (SSL): Learns feature representations using contrastive learning with data augmentations, including Gaussian noise, scaling, time warping, and segment permutation.
	2.	Classifier Models:
	•	Tuned Dense Classifier: A feedforward neural network with dropout regularization.
	•	LSTM Classifier: An LSTM-based network to capture time dependencies in the gesture data.

Performance

Model	Train Accuracy	Validation Accuracy	Test Accuracy
Tuned Classifier	68.75%	66.67%	66.67%
LSTM Classifier	75.62%	74.17%	74.17%

Visualization

The notebook generates visualizations for:
	•	Training Loss over Epochs: Shows the convergence of the Siamese Network during contrastive learning.
	•	2D PCA of Embeddings: Visualizes feature separability across classes in 2D.

Evaluation Metrics

	1.	Accuracy: Overall percentage of correctly classified samples.
	2.	Confusion Matrix: Detailed performance breakdown by class.
	3.	Classification Report: Precision, Recall, and F1-score for each class.

Results and Conclusion

The self-supervised learning approach improved classification accuracy by 7.5% using an LSTM-based classifier compared to traditional methods. This demonstrates the effectiveness of self-supervised learning for time series classification.

Future Work

	•	Experiment with advanced SSL techniques like masked autoencoders.
	•	Explore additional augmentations to improve class separability.

Acknowledgments

Special thanks to:
	•	PyTorch for the machine learning framework.
	•	scikit-learn for utilities in data processing.

