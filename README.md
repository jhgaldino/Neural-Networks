## Neural Networks: Predicting Soccer Results

### Introduction

This project uses neural networks to predict the results of the last round of a soccer championship. Focusing on machine learning techniques, the project aims to offer insights into how different neural network parameters can influence predictions.

### Model Parameters

To explore different results, the following parameters can be adjusted:

hidden_layers (int > 0): Defines the number of hidden layers in the network, influencing the complexity of the model.

epochs (int > 0): Number of model training cycles.

lr (float between 0 and 1): Learning rate, determines the learning speed of the model.

momentum (float between 0 and 1): Contributes to the updating of weights during training, affecting the convergence of the model.

### Data Preparation

The data is initially cleaned, removing unnecessary columns and normalizing values. The dataset is then divided into training and testing parts.

### Model Structure

The model uses a user-specified neural network architecture, including the number of layers and neurons. The network is trained with the input data and adjusted based on the defined parameters.

### Running the Model

This project is a Jupyter Notebook. Run each cell in the notebook to see the process step by step. This includes loading the data, preparing it, defining the model, training it, and visualizing the results.
