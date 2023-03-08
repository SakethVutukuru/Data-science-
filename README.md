# Data-science

## 1- Fake News Detection
An example of how to use scikit-learn to build a machine learning model for text classification. 
Data- news.zip
### Passive Aggressive (PA) Classifier 
is a type of online learning algorithm used for classification tasks. It falls under the category of supervised machine learning algorithms that are used for binary classification problems.
PA classifiers work on the principle of being passive when the current model correctly classifies a training example, but aggressive when it makes a mistake. 
They are particularly useful when the data is large and continuously generated, such as in stream-based applications or in cases where the data is too big to fit in memory. 
The algorithm updates the model parameters as new examples are presented to it, making it an online learning algorithm. 
The PA classifier is implemented in scikit-learn and can be used with various parameters, including maximum step size (regularization), whether the intercept should be estimated or not, and the maximum number of iterations for the algorithm to converge.
### How do we detect Fake news.
-The dataset is loaded into a pandas dataframe and divided into training and testing datasets.
-A TfidfVectorizer object is created, which converts text into numerical representations called TF-IDF vectors, to be used as input data for the machine learning model.
-The TfidfVectorizer object is fit on the training dataset and transforms both the training and testing datasets.
-A PassiveAggressiveClassifier object is created, which is a linear model used for binary classification tasks.
-The PassiveAggressiveClassifier object is fit on the transformed training dataset.
-The accuracy of the model is evaluated on the transformed testing dataset, and the confusion matrix is printed as well to visualize true positives, true negatives, false positives, and false negatives.
-Overall, the code uses machine learning techniques to detect fake news by training a classification model on a dataset of news articles and their labels (whether they are fake or not). The model uses the numerical representations of the text data to make predictions about whether a news article is real or fake, based on the patterns it learned during training.

## 2 - Road Lane-Line Detection with Python & OpenCV
This project implements a lane detection algorithm that can identify and mark the lanes on a road. The project is implemented using Python and OpenCV.
### Frame Masking and Hough Line Transformation
To detect white markings in the lane, first, we need to mask the rest part of the frame. We do this using frame masking. The frame is nothing but a NumPy array of image pixel values. To mask the unnecessary pixel of the frame, we simply update those pixel values to 0 in the NumPy array.
After making we need to detect lane lines. The technique used to detect mathematical shapes like this is called Hough Transform. Hough transformation can detect shapes like rectangles, circles, triangles, and lines.

## 3- Sentiment Analysis model using R programming.
What is sentiment Analaysis?

Sentiment analysis, also known as opinion mining or emotion AI, is a process that involves using natural language processing (NLP), text analysis, and computational linguistics to identify, extract, and quantify the affective states and subjective information in a given piece of text. The purpose of sentiment analysis is to determine whether the emotional tone of a message is positive, negative, or neutral.

Sentiment analysis combines machine learning and NLP to analyze a piece of text and determine the sentiment behind it. It can be used to analyze various types of text data, including emails, customer support chat transcripts, social media comments, and reviews.
### How do we find the sentiments? Which libraries do we use?
Three general-purpose lexicons, such as AFINN and Bing Loughran, will be used.
The unigrams are used in these three lexicons. A word selected from a given body of text is all that makes up a sequence of one item, or a unigram, which is a form of n-gram model. The words are given scores in the AFINN lexicon model that range from -5 to 5. A rise in negativity matches a decline in sentiment, whereas a rise in positivity matches a rise in sentiment. On the other hand, the Bing Lexicon Model categorizes the sentiment as either positive or negative. Finally, there is the Loughran model, which analyzes the shareholder reports. To extract the sentiments for this project, we will use the Bing lexicons.
<img width="967" alt="image" src="https://user-images.githubusercontent.com/115586733/223658698-a510b7ba-9e19-4bed-b23b-f23f6087bea9.png">

# Contributing
If you find any bugs or issues with any of these projects, please feel free to submit an issue or a pull request. I welcome contributions from the community.
