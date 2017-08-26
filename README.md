# NaiveBayesClassifier

Made by Chris Kormaris

Programming Language: Python


Unzip the compressed file *"LingspamDataset.zip"* in the same directory where the Python files are. 700 train documents and 260 test documents will reside inside the uncompressed folder.

## Feature Selection with Information Gain

Let's begin by denoting the variable C, which takes the values: C=1 for spam documents and C=0 for ham documents.
First, run the python file "FeatureSelectionUsingIG.py" to generate the features tokens that we'll use. Run:
```python
python FeatureSelectionUsingIG.py
```
Feature selection for the most useful feature tokens, using Information Gain (IG) has been implemented in this program. The feature tokens are boolean, as in they take boolean values, 0 if the token does not appear in a text or 1 if the token appears in a text. The boolean values are assigned while generating the feature vectors of each text. At the start of the program, all the train files of the corpus are being parsed and we count in how many spam or ham documents in total, each word appears. The results are being saved in dictionary data structures with the names: *"feature_spam_frequency"*, *"feature_ham_frequency"* and *"feature_frequency"* respectively, with feature tokens being the keys and and frequencies being the values. These dictionary variables are used for the calculation of the probabilities of the Information Gain algorithm. We calculate the entropy *H(C)* and print it. We proceed by adding each candidate feature token to the "IG" dictionary, while calculating its IG score. Finally, we select the top *m* feature tokens, on terms of the highest Information Gain (IG) scores. The Information Gain score of each feature token is calculated using the formula:

![Information Gain](http://latex.codecogs.com/gif.latex?IG%28X%20%2C%20C%29%20%3D%20IG%20%28C%20%2C%20X%29%20%3D%20H%28C%29%20-%20%5Csum_%7Bi%3D0%7D%5E%7B1%7D%20%7BP%20%28X%3Di%29%20%5Ccdot%20H%20%28C%7CX%3Di%29%7D)

where i=0 and i=1 are the boolean values that a feature token may take, indicating if it appears or not in a text.
Concretely, the feature X that reduces the entropy less is the most desired candidate feature because it can discriminate the category of a document more efficiently. The number of feature tokens that the feature selection algorithm returns is set to 1000. The number of feature tokens to use depends on the classification task we want to execute. For our Naive-Bayes spam-ham classifier a number of 1000 feature tokens is a good choice.

In addition, there has been implemented, inside block comments, an alternative way for calculating the Information Gain score, by using the following formula:

![Information Gain alterative](http://latex.codecogs.com/gif.latex?IG%28X%2C%20C%29_%7Balt%7D%20%3D%20IG%28C%2C%20X%29_%7Balt%7D%20%3D%20%7CP%28X%3D1%7CC%3D0%29%20-%20P%20%28X%3D1%7CC%3D1%29%7C)

which is the absolute difference between the conditional probabilities for the 2 classes (spam or ham). The tokens for which this absolute difference is bigger are selected as feature tokens of the feature vectors, based on which we will classify our corpus files. Using the feature tokens of this formula, slightly worse accuracy has been observed. In the end of the program "FeatureSelectionWithIG.py", two dictionaries will be created, which contain the feature tokens that will be used for the Naive-Bayes classifier and they are saved in the files: *"spam_feature_dictionary.txt"* and *"ham_feature_dictionary.txt"*.

## Perform Naive-Bayes classification

After the feature selection step, run *"NaiveBayesClassifier.py"* to start the classification process. Run:
```python
python NaiveBayesClassifier.py
```
First, the classifier counts in how many spam documents each spam feature token is found and in how many ham documents each ham feature token is found. We use boolean features. The total frequency of each spam and ham feature token is saved in two dictionaries: "spam_feature_tokens_frequencies" and "ham_feature_tokens_frequencies". We also count the number of spam and ham train documents and the total number of train documents. We use the acquired frequencies to estimate the Laplace probabilities that will determine the correct category of the test data. The Laplace probability estimate classification calculates the conditional probability of each test document's feature vector for each 2 categories (spam or ham) and categorizes the document in the class for which the conditional probability is higher. I.e. for the feature vector X<sub>i</sub> of the test document *i* we calculate the probabilities p(X<sub>i</sub>|C=1) and p(X<sub>i</sub>|C=0) (reminder: C=1 for spam, C=0 for ham). The class that has higher probability will be the predicted class of the train document. For the calculation of the probabilies we also use Laplace smoothing, adding 1 in the enumerator and 2 (for the 2 classes spam and ham) in the denominator. This is the formula that is used for calculating the probability of the feature token i, belonging to a spam document:

![Laplace Smoothing token](http://latex.codecogs.com/gif.latex?\frac{spamDocumentFrequencyOfToken[i]%20&plus;%201}%20{numberOfSpamDocuments%20&plus;%20numberOfFeatures}%20%3D%20\frac{spamDocumentFrequencyOfToken[i]%20&plus;%201}%20{numberOfSpamDocuments%20&plus;%20|V|})

*(where |V| is the is the size of the dictionary of feature tokens)*

To calculate the probability of the entire feature vector belonging to the spam class we multiply the probability of each separate feature token belonging to the spam class. The exact formula is:

![Laplace Smoothing vector](http://latex.codecogs.com/gif.latex?probOfFeatureVectorBelongingToSpam%20%3D%20\frac{P(C%3D1)}{P(featureVector)}%20\cdot%20\prod_i%20\frac{spamDocumentFrequencyOfToken[i]%20&plus;%201}%20{numberOfSpamDocuments%20&plus;%20|V|})

We do the same for the ham class. We can omit the term *P(featureVector)* since it's the same for both 2 classes. The probability of the 2 which is bigger, indicates the category that the given test document and its feature vector is more likely to belong to. Also, it is a good idea to use the numerically stable "logsumexp trick", thus taking sum of exponentials, rather than multiplications of probabilities.


The execution results of the classifier delivered an accuracy of **98.08 %**, while using 1000 features tokens and **97.69 %**, while using 100 features tokens.

**Notes**

* Statistics show that the Naive-Bayes classifier has a higher accuracy on a small corpus, rather than a big corpus.
* Console execution results are included in the "console_outputs" folder.

