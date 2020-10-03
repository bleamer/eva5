

# Data Augmentation | GradCAM
This notebook implements CIFAR10 classification using ResNet NN architecture using PyTorch,
and uses GradCAM analysis for model explanation  to explain which layer of model was activated for the give image

Here we use Data Augmentation implementusing [Albumentations](https://github.com/albumentations-team/albumentations) library

The results with using ResNet18 architecture with (2,2,2,2), that is, 4 residual blocks of size 2 each.
 
----
| Attribute | Value |
|:--- | :--- |
| Test Accuracy | 87.78 |
| Trained Parameter count   |11173962 |
| Dropout | 0.1 |
| Total Epochs | 25 |
| Batch Size | 64|

----

![](Accuracies.png)

The model uses:
* 3x3 Convolution
* Batch Normalization
* Max Pooling
* 4 X Residual networks of size 2 

To run 

You can install dependencies using  
`$ pip install -r req.txt`


###Grad CAM Images

####Correct Images </br> ![](Correct_grad_cam_1.png)![](Correct_grad_cam_2.png)![](Correct_grad_cam_3.png)![](Correct_grad_cam_4.png)

-----

####Incorrect Images </br> ![](Incorrect_grad_cam_1.png)![](Incorrect_grad_cam_2.png)![](Incorrect_grad_cam_3.png)![](Incorrect_grad_cam_4.png) </br> ![](Incorrect_grad_cam_5.png)![](Incorrect_grad_cam_6.png)![](Incorrect_grad_cam_7.png)![](Incorrect_grad_cam_8.png)