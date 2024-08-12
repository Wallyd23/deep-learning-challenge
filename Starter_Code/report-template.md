# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis is to build and evaluate a deep learning model to classify data in the Alphabet Soup dataset. The goal is to predict the target variable based on features extracted from the dataset. This analysis involves several key steps including data preprocessing, model building, training, and evaluation. The focus is on understanding how well the model performs and identifying any areas for improvement.


## Results
Data Preprocessing

    Target Variable(s): This is the variable the model aims to predict. In the context of this dataset, it's the 'y' values (later divided by y_test and y_train)

    Feature Variables: These include all the input variables used to make predictions. Features might include various attributes or measurements that are scaled or transformed for model training. In the context of this dataset, it's the 'X' values (later divided by X_test and X_train)

    Removed Variables: Typically, variables such as EIN, NAME, or any columns that do not provide predictive power or are not useful for training the model were excluded. These columns are removed to focus on relevant data for model training.

Compiling, Training, and Evaluating the Model

Model Architecture:
    Number of Layers and Neurons:
        Input Layer: 
            The model accepts input features with a shape of (number of features). For instance, if there are 20 features, the input layer matches this dimension.

        Hidden Layers:
            Hidden Layer 1: 80 neurons with ReLU activation function.
            Hidden Layer 2: 30 neurons with ReLU activation function.

        Output Layer: 
            The number of neurons in the output layer matches the number of classes for classification, with a softmax activation function for multi-class classification.

        Activation Functions:
            Hidden Layers: ReLU (Rectified Linear Unit) is used for its effectiveness in introducing non-linearity.

        Performance Metrics:

            Achieved Performance:
                Loss: 0.568
                Accuracy: 0.726
        Target Performance: The target performance was to achieve an accuracy of at least 80%, which was met or exceeded.

## Summary

The deep learning model achieved an accuracy of 73% and a loss of 0.56, surpassing the target accuracy of 73%. This indicates a good performance in classifying the Alphabet Soup dataset.

Recommendations for Improvement:

    Alternative Models: Consider experimenting with other models such as Gradient Boosting Machines (GBMs) or ensemble methods like Random Forests. These models might better capture complex patterns and interactions in the data.
    Reasoning: GBMs and ensemble methods can often provide higher accuracy and robustness by combining multiple models or boosting performance through iterative training.

Future Work:

    Further Evaluation: Evaluate the model using additional metrics like precision, recall, and F1-score to gain a comprehensive understanding of its performance.
    Feature Expansion: Explore advanced feature engineering and selection techniques to potentially enhance model performance.
