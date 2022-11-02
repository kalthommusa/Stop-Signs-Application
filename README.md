# Objective

Design, train, and test deep convolutional neural networks with transfer learning to perform stop signs classification task with PyTorch and IBM Computer Vision Studio.

----------------------------------------------------------

# CV Studio

When building Computer Vision applications, we deal with a lot of images, videos and other assets that require storage. Image processing also requries significant computing power. The ultimate goal of every computer vision project is to have it deployed as part of an application and every application requires infrastructure to run.

CV Studio is a Computer Vision learning tool for building, running, and managing AI Vision computer models and optimizing decisions anywhere on the IBM Cloud. It uses cutting-edge tools like Jupyter Notebook, Watson Machine Learning, Elyra, and more.

----------------------------------------------------------

# Usge

Simply enter the URL of the web app below into any web browser, and enjoy testing the model:

`
https://stop-not-stop-app-635dc9a1b6ccf16d413f30f6.u8wpmjcrkw2.eu-gb.codeengine.appdomain.cloud/
`

You can upload or drop & drag an image with or without a stop sign and the classifier will classify it.

![misclassified stop-3](images/the-app.png)

----------------------------------------------------------

# Project steps / pipline:

  * 1- Built the datast

  Initily I uploaded a 200 images on CV Studio

  * 2- Annotate the dataset

  Here I created two classes or labels: 
  stop
  not-stop
  then divided and labeled the uploaded images as a 100 images of each class

  * 3- Train the model

  Develop & train a model by creating a new training run on CV Studio.


  Used a Jubytur notbook on CV Studio to design and train a custom classifier 

  using `ResNet` pretrained model archtecther as a fixed feature extractor to perform a Supervised Machine Learning Task  

  deploy the custom model on IBM CV Studio to make predictions on new images.


  * 4- Test the model


labeled in the Annotation tab of CV Studio.

# Results & Observation

The Overall performance of the classifier seems pretty good on the test data, the classifier was able to classify all of the stop test images correctly as well as the majority of the not-stop test images.

![correct stop-1](images/correct-stop-1.png)

![correct stop-2](images/correct-stop-2.png)

![correct stop-3](images/correct-stop-3.png)

![correct stop-4](images/correct-stop-4.png)

![correct not-stop-1](images/correct-not-stop-1.png)
![correct not-stop-2](images/correct-not-stop-2.png)
![correct not-stop-4](images/correct-not-stop-4.png)

However, it fails to recognize the not-stop below correctly and classified as a stop one.
![misclassified not-stop-3](images/misclassified-not-stop-3.png) 

Moreover, I have tested the app with more unseen imges and catched these misclassied results by the classifier.

![misclassified stop-1](images/misclassified-stop1.png)

![misclassified stop-2](images/misclassified-stop2.png)

![misclassified stop-3](images/misclassified-stop3.png)


# Re-train The Model

Machine learning workflow is an iterative process, and it is preety common when develop a machine learning model to take a step/s back and adjust some steps in order to improve the performace of the classifier and make it produce better peridctions.


My custom deep nueral network made some mistakes peridcting the wrong class of some of the test images due to the lack of the number of the training dataset, one way to increase the performance of the classifier is to increase the size of the training dataset.

Therefore I gathured more images, uploaded and annotated them on CV Studio. I have doubled the size of the training dataset,so my new dataset has 400 images divided into 200 images for each class.

![](images/400-uploaded-images.png)

![](images/400-annotated-images.png)


Then I retranied the deep neural network on the updated dataset. And it terns out, incraesing the number of images leads to increase the accuracy of the custom model, as well as the overall trainig run time. 

![accuracy & trainig run time bafore adding more images](images/accuracy-200.png)


![accuracy & trainig run time after adding more images](images/accuracy-400.png)


By feeding the classifier more data it get to learn more information about the stop & not-stop sings and understand the relationship between the input and the output, as a result it becoms more smarter and gave better peridctions.

As we can see in the images below, the deep neural network was able to correctly clasify the prefously misclassifief images, which indicats improved performance.

![accuracy & trainig run time after adding more images](images/correct-misclassified-stop-3.png)

![accuracy & trainig run time after adding more images](images/correct-misclassified-stop1.png)

![accuracy & trainig run time after adding more images](images/correct-misclassified-stop2.png)

![accuracy & trainig run time after adding more images](images/correct-misclassified-stop3.png)

