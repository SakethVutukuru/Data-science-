# Data-science

## 1- Fake News Detection
An example of how to use scikit-learn to build a machine learning model for text classification.
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
