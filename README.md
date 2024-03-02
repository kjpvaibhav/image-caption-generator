# image-caption-generator

## What is Image Caption Generator?
Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.

## Image Caption Generator with CNN – About the Python based Project
The objective of this project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.

In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

## The Dataset of Python based Project
For the image caption generator, we will be using the Flickr_8K dataset. There are also other big datasets like Flickr_30K and MSCOCO dataset but it can take weeks just to train the network so we will be using a small Flickr8k dataset. The advantage of a huge dataset is that we can build better models.

Thanks to Jason Brownlee for providing a direct link to download the dataset (Size: 1GB).
* Flicker8k_Dataset - https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
* Flickr_8k_text - https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
The Flickr_8k_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions separated by newline(“\n”).

## Pre-requisites
This project requires good knowledge of Deep learning, Python, working on Jupyter notebooks, Keras library, Numpy, and Natural language processing.

## Make sure you have installed all the following necessary libraries:
Using pip install 
tensorflow
keras
pillow
numpy
tqdm
jupyterlab

## What is CNN?
Convolutional Neural networks are specialized deep neural networks which can process the data that has input shape like a 2D matrix. Images are easily represented as a 2D matrix and CNN is very useful in working with images.
CNN is basically used for image classifications and identifying if an image is a bird, a plane or Superman, etc.
It scans images from left to right and top to bottom to pull out important features from the image and combines the feature to classify images. It can handle the images that have been translated, rotated, scaled and changes in perspective.

## What is LSTM?
LSTM stands for Long short term memory, they are a type of RNN (recurrent neural network) which is well suited for sequence prediction problems. Based on the previous text, we can predict what the next word will be. It has proven itself effective from the traditional RNN by overcoming the limitations of RNN which had short term memory. LSTM can carry out relevant information throughout the processing of inputs and with a forget gate, it discards non-relevant information.

## Image Caption Generator Model
So, to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.

* CNN is used for extracting features from the image. We will use the pre-trained model Xception.
* LSTM will use the information from CNN to help generate a description of the image.

## Project File Structure
Downloaded from dataset:
* Flicker8k_Dataset – Dataset folder which contains 8091 images.
* Flickr_8k_text – Dataset folder which contains text files and captions of images.

The below files will be created by us while making the project.
* Models – It will contain our trained models.
* Descriptions.txt – This text file contains all image names and their captions after preprocessing.
* Features.p – Pickle object that contains an image and their feature vector extracted from the Xception pre-trained CNN model.
* Tokenizer.p – Contains tokens mapped with an index value.
* Model.png – Visual representation of dimensions of our project.
* Testing_caption_generator.py – Python file for generating a caption of any image.
* Training_caption_generator.py – Jupyter notebook in which we train and build our image caption generator.

Download files - https://drive.google.com/open?id=13oJ_9jeylTmW7ivmuNmadwraWceHoQbK

## Building the Python based Project
Let’s start by initializing the jupyter notebook server by typing jupyter lab in the console of your project folder. It will open up the interactive Python notebook where you can run your code. Create a Python3 notebook and name it training_caption_generator.py

1. First, we import all the necessary packages
   
2. Getting and performing data cleaning
The main text file which contains all image captions is Flickr8k.token in our Flickr_8k_text folder.
The format of our file is image and caption separated by a new line (“\n”).
Each image has 5 captions and we can see that #(0 to 5)number is assigned for each caption.
We will define 5 functions:
* load_doc( filename ) – For loading the document file and reading the contents inside the file into a string.
* all_img_captions( filename ) – This function will create a descriptions dictionary that maps images with a list of 5 captions.
* cleaning_text( descriptions) – This function takes all descriptions and performs data cleaning. This is an important step when we work with textual data, according to our goal, we decide what type of cleaning we want to perform on the text. In our case, we will be removing punctuations, converting all text to lowercase and removing words that contain numbers.
So, a caption like “A man riding on a three-wheeled wheelchair” will be transformed into “man riding on three wheeled wheelchair”
* text_vocabulary( descriptions ) – This is a simple function that will separate all the unique words and create the vocabulary from all the descriptions.
* save_descriptions( descriptions, filename ) – This function will create a list of all the descriptions that have been preprocessed and store them into a file. We will create a descriptions.txt file to store all the captions.

3. Extracting the feature vector from all images 
This technique is also called transfer learning, we don’t have to do everything on our own, we use the pre-trained model that have been already trained on large datasets and extract the features from these models and use them for our tasks. We are using the Xception model which has been trained on imagenet dataset that had 1000 different classes to classify. We can directly import this model from the keras.applications . Make sure you are connected to the internet as the weights get automatically downloaded. Since the Xception model was originally built for imagenet, we will do little changes for integrating with our model. One thing to notice is that the Xception model takes 299*299*3 image size as input. We will remove the last classification layer and get the 2048 feature vector.
model = Xception( include_top=False, pooling=’avg’ )
The function extract_features() will extract features for all images and we will map image names with their respective feature array. Then we will dump the features dictionary into a “features.p” pickle file.
This process can take a lot of time depending on your system. I am using an Nvidia 1050 GPU for training purpose so it took me around 7 minutes for performing this task. However, if you are using CPU then this process might take 1-2 hours. You can comment out the code and directly load the features from our pickle file.

5. Loading dataset for Training the model
In our Flickr_8k_test folder, we have Flickr_8k.trainImages.txt file that contains a list of 6000 image names that we will use for training.
For loading the training dataset, we need more functions:
* load_photos( filename ) – This will load the text file in a string and will return the list of image names.
* load_clean_descriptions( filename, photos ) – This function will create a dictionary that contains captions for each photo from the list of photos. We also append the <start> and <end> identifier for each caption. We need this so that our LSTM model can identify the starting and ending of the caption.
* load_features(photos) – This function will give us the dictionary for image names and their feature vector which we have previously extracted from the Xception model.

5. Tokenizing the vocabulary 
Computers don’t understand English words, for computers, we will have to represent them with numbers. So, we will map each word of the vocabulary with a unique index value. Keras library provides us with the tokenizer function that we will use to create tokens from our vocabulary and save them to a “tokenizer.p” pickle file.
Our vocabulary contains 7577 words.
We calculate the maximum length of the descriptions. This is important for deciding the model structure parameters. Max_length of description is 32.

6. Create Data generator
Let us first see how the input and output of our model will look like. To make this task into a supervised learning task, we have to provide input and output to the model for training. We have to train our model on 6000 images and each image will contain 2048 length feature vector and caption is also represented as numbers. This amount of data for 6000 images is not possible to hold into memory so we will be using a generator method that will yield batches.
The generator will yield the input and output sequence.
For example:
The input to our model is [x1, x2] and the output will be y, where x1 is the 2048 feature vector of that image, x2 is the input text sequence and y is the output text sequence that the model has to predict.
x1(feature vector)	x2(Text sequence)	              y(word to predict)
feature	            start,	                              two
feature            	start, two                            dogs
feature	            start, two, dogs                      drink
feature	            start, two, dogs, drink               water
feature	            start, two, dogs, drink, water	      end

7. Defining the CNN-RNN model
To define the structure of the model, we will be using the Keras Model from Functional API. It will consist of three major parts:
* Feature Extractor – The feature extracted from the image has a size of 2048, with a dense layer, we will reduce the dimensions to 256 nodes.
* Sequence Processor – An embedding layer will handle the textual input, followed by the LSTM layer.
* Decoder – By merging the output from the above two layers, we will process by the dense layer to make the final prediction. The final layer will contain the number of nodes equal to our vocabulary size.

8. Training the model
To train the model, we will be using the 6000 training images by generating the input and output sequences in batches and fitting them to the model using model.fit_generator() method. We also save the model to our models folder. This will take some time depending on your system capability.

9. Testing the model
The model has been trained, now, we will make a separate file testing_caption_generator.py which will load the model and generate predictions. The predictions contain the max length of index values so we will use the same tokenizer.p pickle file to get the words from their index values.

## Results
* img1.jpg - man is standing on rock overlooking the mountains
* img2.jpg - man in yellow kayak is reflecting up river
* img5.jpg - two girls are playing in the grass
