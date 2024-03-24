# Digit Recognition

This project offers diverse solutions for the digit recognition problem, exploring various machine learning algorithms and techniques.

## Requirements

- [Python][Python] 3.10.4
- [Pip][Pip] 22.0.4
- [Numpy][numpy] 1.26.4
- [Pandas][pandas] 2.2.1
- [TensorFlow][tensorflow] 2.16.1
- [TensorFlow Decision Forests][tensorflow-decision-forests] 1.9.0

[Python]: https://www.python.org/downloads/release/python-3104/
[Pip]: https://pypi.org/project/pip/
[numpy]: https://pypi.org/project/numpy/
[pandas]: https://pypi.org/project/pandas/
[tensorflow]: https://pypi.org/project/tensorflow/
[tensorflow-decision-forests]: https://pypi.org/project/tensorflow-decision-forests/

## Approaches

### Neural Network Implementation

The project contains a neural network implementation based on Tensorflow (Keras) libraries specifically designed for digit recognition tasks (`neural_network.py`). By default it provides solution with 2 dense layers for basic MNIST data but it can also create model with convolution and pooling layer in case of images with larger size.

In addition to library solutions, this project offers a self-implemented neural network. Speed is surely lower but it can present what is actually going on in each of neuron and how parameters created network are modified. It can solve other problems, like sign of multiplication of two integer from $[-1,1]$ (implemented in `custom_network/main.py`) or more complicated problems by passing appropriate parameters to $Network$ class (`custom_network/network.py`).

### Decision Forest Algorithm

Another approach featured in this project is the decision forest algorithm. It operates by constructing multiple decision trees during the training phase. Each decision tree in the forest is trained independently on a subset of the training data and using a random selection of features. During classification, the input image is passed through each decision tree, and the final prediction is made based on the majority vote of all decision trees.

This solution can efficiently reduce importance of pixels which are not used in digit recognition, especially those placed on corners. It is also less sensitive for overfitting, especially when number of independent decition trees is high enough. It is worth noting that the decision forest may exhibit a slightly lower accuracy compared to neural network, because on average, the recognition rate may be a few percentage points lower.

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm utilized in this project for digit recognition tasks. It assumes that there is no training data and the goal is to separate different images into sets. It groups similar digit images together based on their feature representations, aiding in data preprocessing and organization prior to model training.

This clustering process serves to enhance the quality of the dataset by isolating clusters of digit images that share common characteristics, thereby mitigating the influence of noise and outliers. While DBSCAN showed limited success in digit recognition, its utility extends to images with more intricate details that distinctly define their presence within a given cluster.

## Contact

For inquiries, support, or collaboration opportunities, please contact:

- krzysztof.urbisz.ku@gmail.com
- https://github.com/kurbisz/
