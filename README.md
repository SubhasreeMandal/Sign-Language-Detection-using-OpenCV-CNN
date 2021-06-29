# Sign-Language-Detection-using-OpenCV-CNN

 **Software Requirement Specification**	
 
 The prerequisites software & libraries for the sign language project are:
 
•	Python (2.7)

•	IDE (Jupyter)

•	Numpy (version 1.16.5)

•	cv2 (openCV) (version 3.4.2)

•	Tensorflow

•	CNN model

**Steps**
  
  This is divided into 6 parts:
  
1.	Image Collection
2.	Image Preprocessing
3.	Pre-training using CNN model
4.	Training and testing the model
5.	Evaluation
6.	Predicting the hand gestures

**Implementation of these steps**

**1.	Image collection:**

The dataset consists of 15300 images with 1700 images per category. Among which, 12500 images are used for training purpose and 3800 images for testing purpose. 
For creating the dataset, we have a live feed from the video camera and every frame that detects a hand in the ROI (region of interest) created is saved in a directory that contains two folders train and test, each containing 10 folders containing images captured using the create_gesture_data.py. Then the images undergo a series of processing.

![image](https://user-images.githubusercontent.com/48086440/123749304-e57f7780-d8d2-11eb-9eb5-5e0b49af840d.png)



**2.  Image Preprocessing:**

Now for creating the dataset we get the live cam feed using OpenCV and created an ROI that is nothing but the part of the frame where we want to detect the hand in for the gestures. Then backgrounds are detected and eliminated using opencv methods.

	cv2.accumulateWeighted() detect active objects from the difference obtained from the reference frame and the current frame. Reference frame is detected by giving 5 seconds break before capturing.

	cv2.absdiff() finds the absolute difference between pixels of two image array that is the background and the hand image. Then it extracts just the pixels of the object.

	cv2.threshold() then separate the hand image from the background.



![image](https://user-images.githubusercontent.com/48086440/123748009-4908a580-d8d1-11eb-92ea-8163c2d90115.png)



Images are captured in RGB colorspaces. It becomes more difficult to segment the hands from the RGB colorspace. That’s why, those images are processed with opencv methods.


![image](https://user-images.githubusercontent.com/48086440/123748180-7ead8e80-d8d1-11eb-8de8-d8c7cc621a68.png)


![image](https://user-images.githubusercontent.com/48086440/123748201-866d3300-d8d1-11eb-8d66-4dc1834560e7.png)



**3.  Creation of CNN model:**

**Pre-training:** 

Transfer learning is used here. Vgg16 is a convolution neural network that is used to pre-train the image dataset. It is easily downloadable from Keras API. It is a pre-trained architecture that can detect generic visual features present in our sign language dataset. We also have imported “preprocessing function” to normalize the data before training that is to change raw feature vector to suitable representation for estimators.

We use the pre-trained model's (vgg16) architecture to create a new dataset from our input images in this approach. We'll import the Convolutional and Pooling layers but leave out the "top portion" of the model (the Fully-Connected layer).

We'll pass our images through VGG16's convolutional layers, which will output a Feature Stack of the detected visual features. Then it is ready for forward propagation (input layerhidden layerfully connected layer) in CNN model.

In this way, training and testing datasets are prepared. First, we load the data using ImageDataGenerator of keras through which we can use the flow_from_directory() function to load the train and test set data, and each of the names of the number folders will be the class names for the images loaded.


**Power of VGG16 model:**

Vgg16 is a convolutional neural network that was proposed by Oxford visual geometry group. It is considered as one of the excellent vision model architecture till date. Most unique thing about Vgg16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used same padding and maxpool layer of 2x2 filter of stride 2.

we think, if we didn’t use this model for pre-training the dataset, then only CNN sequential model couldn’t give this higher accuracy. CNN along with pre-training gives the expected accuracy, precision and recall.


**CNN model:**

Now on the pre-trained dataset, a CNN model is implemented. PlotImages () function is for plotting the feature map of the images.


![image](https://user-images.githubusercontent.com/48086440/123748392-c46a5700-d8d1-11eb-92d4-2d539ed5aa8b.png)


Now we design the CNN as follows. we'll be creating a Keras Model with the Sequential model API.
It contains two part—

	Feature extraction

	Classification


![image](https://user-images.githubusercontent.com/48086440/123748444-d8ae5400-d8d1-11eb-8927-8b9790fd2216.png)



**Feature extraction**

Convo layer apply a filter to an input to create a feature map that summarizes the presence of detected features in the input. Conv2D layer calculates the feature map for two-dimensional convolutional layer in a convolutional neural network.

The output volume of the Conv. layer is fed to an elementwise activation function, commonly a Rectified-Linear Unit (ReLu). A ReLu function will apply a max (0, x) function, thresholding at 0. The dimensions of the volume are left unchanged.

Pooling layers provide an approach to down sampling feature maps by summarizing the presence of features in patches of the feature map. Here max pooling MaxPool2D is used that summarize the most activated presence of a feature.        


![image](https://user-images.githubusercontent.com/48086440/123748481-e8c63380-d8d1-11eb-9f58-df26902b66ca.png)



**Classification:**

The output volume, i.e. 'convolved features,' are passed to a Fully-Connected Layer of nodes. Like conventional neural-networks, every node in this layer is connected to every node in the volume of features being fed-forward.

We call the Flatten () method at the start of the Fully-Connected Layer. This is to transform the 3-Dimensional feature maps into a 1-Dimensional input tensor.

We'll construct a Fully-Connected layer using Dense layers.

Since this is a multi-class problem, our Fully-Connected layer has a Softmax activation function.

Drop out is also employed.
 It prevents over-fitting.
 

![image](https://user-images.githubusercontent.com/48086440/123748564-fed3f400-d8d1-11eb-862a-9107a9295698.png)


![image](https://user-images.githubusercontent.com/48086440/123748583-05626b80-d8d2-11eb-864c-4f1013e26f64.png)

   
 ![image](https://user-images.githubusercontent.com/48086440/123748693-2925b180-d8d2-11eb-8ff4-8b2c0d344f1a.png)



**4.  Training and testing the CNN model**


**Training:**

Keras models include the ability to interact automatically with the model during training. Here call back technique EarlyStopping is used.

Keras deep learning library Adam (Adaptive Moment Estimation) is used here with learning rate 0.001. First, an instance of the class must be created and configured, then specified to the “optimizer” argument when calling the fit () function on the model.

Next, deep learning neural networks are trained using the stochastic gradient descent algorithm (SGD).

Stochastic gradient descent is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the mode.


![image](https://user-images.githubusercontent.com/48086440/123748757-4195cc00-d8d2-11eb-8cec-4a08be7ffdcd.png)



After every epoch, the accuracy and loss are calculated using the validation dataset and if the validation loss is not decreasing, the LR of the model is reduced using the Reduce LR to prevent the model from overshooting the minima of loss. 

The EarlyStopping algorithm works so that if the validation accuracy keeps on decreasing for some epochs then the training is stopped.
It monitors the performance of the model and stopping the training process prevents overtraining.


![image](https://user-images.githubusercontent.com/48086440/123748800-51151500-d8d2-11eb-8023-e33d56920502.png)


**Testing:**

Testing labels--


![image](https://user-images.githubusercontent.com/48086440/123748847-612cf480-d8d2-11eb-9d53-4038f295961f.png)


**5.  Evaluation**

After testing our model with our dataset, we received 84% accuracy, 85% precision, 83% recall and loss is 8%. Hence we can, see great accuracy with low loss indicates a best case that contains low error on few data. 


![image](https://user-images.githubusercontent.com/48086440/123748936-802b8680-d8d2-11eb-9991-7138ab48c86b.png)

![image](https://user-images.githubusercontent.com/48086440/123748968-87529480-d8d2-11eb-9730-4d1868204889.png)

![image](https://user-images.githubusercontent.com/48086440/123749029-9a656480-d8d2-11eb-95d5-079decf594f3.png)


**6.  Predicting hand gestures**

In this, we have created a bounding box for detecting the ROI and calculated the accumulated_weight as we did in creating the dataset. This is done for identifying any foreground object.

Now we found the max contour and if contour is detected that means a hand is detected so the threshold of the ROI is treated as a test image.

We load the previously saved model using keras.models.load_model and feed the threshold image of the ROI consisting of the hand as an input to the model for prediction.

Few predictions:


![image](https://user-images.githubusercontent.com/48086440/123749129-b537d900-d8d2-11eb-9959-4a3e82f97d28.png)


![image](https://user-images.githubusercontent.com/48086440/123749153-bc5ee700-d8d2-11eb-9dc6-5a8a98484691.png)


![image](https://user-images.githubusercontent.com/48086440/123749178-c54fb880-d8d2-11eb-9a86-c2a92bd07a77.png)















