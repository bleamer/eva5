### MNIST Classification using CNN in PyTorch

[MNIST](http://yann.lecun.com/exdb/mnist/ "MNIST") is an image dataset with collection of handwritten digits.
This attached Jupyter notebook  builds a Convolution Neural Network to read the images in the data and identify the digit.

The model was trained with following configuration:


.. | ..  
--- | ---  
**Trained Parameter Count**|11044
**Max. no. of channels**|16
**Trained for Epochs**|18
**Batch Size**|128
**Loss function**| Negative Logarithmic Loss

The test and train accuracy plots can be seen inside the notebook execution logs.

The notebook defines the Neural Network  using 7 Convolution layer divided into 3 blocks, similar to Resnet architecture barring use of Max Pool layers to reduce the channel size.

Further `Train` and `Test` methods define batching and optimizing using Stochastic Gradient Descent based backpropagation of Loss to Model weights

In the end the graph shows training / test loss and model prediction accuracy.
