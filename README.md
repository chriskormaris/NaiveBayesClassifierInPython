# NaiveBayesClassifier

Made by Chris Kormaris

Programming Language: Python

Unzip the files *"TRAIN.zip"* and *"TEST.zip"*, in the same directory where the Python files are. 8512 train documents and 3000 documents are contained inside these files.

## Feature Selection with Information Gain

Let's begin by denoting the variable C, which takes the values: C=1 for spam documents and C=0 for ham documents.
First, run the python file "FeatureSelectionUsingIG.py" to generate the output file "feature_dictionary.txt", containing the features tokens that we'll use. Run:
```python
python FeatureSelectionUsingIG.py
```
Feature selection for the most useful tokens, using Information Gain (IG) has been implemented in this program. The feature tokens are boolean, as in they take boolean values, 0 if the token does not appear in a text or 1 if the token appears in a text. The boolean values are assigned while generating the feature vectors of each text. At the start of the program, all the train files of the corpus are being parsed and we count in how many spam or ham documents in total, each word appears. The results are being saved in dictionary data structures with the names: *"feature_spam_frequency"*, *"feature_ham_frequency"* and *"feature_frequency"* respectively, with feature tokens being the keys and and frequencies being the values. These dictionary variables are used for the calculation of the probabilities of the Information Gain algorithm. Then, we calculate the entropy H(C) calculated and print it. The information gain for each score for each feature token is calculated using the formula:

![Information Gain](http://latex.codecogs.com/gif.latex?IG%28X%20%2C%20C%29%20%3D%20IG%20%28C%20%2C%20X%29%20%3D%20H%28C%29%20-%20%5Csum_%7Bi%3D0%7D%5E%7B1%7D%20%7BP%20%28X%3Di%29%20%5Ccdot%20H%20%28C%7CX%3Di%29%7D)

where i=0 and i=1 are the boolean values that a feature token may take, indicating if it appears or not in a text.
Concretely, the feature X that reduces the entropy less is the most desired candidate feature because it can discriminate the category of a document more efficiently. The number of feature tokens the feature selection algorithm returns is set to 1000, alternatively 100. The number of feature tokens to use depends on the classification algorithm that we'll use. Since we are building a Naive-Bayes classifier we set the number of features to 100. Using less features is more likely to make better predictions if Naive-Bayes is the classifier of our choice.

In addition, there has been implemented, inside block comments, an alternative way for calculating the Information Gain score, by using the following formula:

![Information Gain alterative](http://latex.codecogs.com/gif.latex?IG%28X%2C%20C%29_%7Balt%7D%20%3D%20IG%28C%2C%20X%29_%7Balt%7D%20%3D%20%7CP%28X%3D1%7CC%3D0%29%20-%20P%20%28X%3D1%7CC%3D1%29%7C)

which is the absolute difference between the conditional probabilities for the 2 classes (spam or ham). The tokens for which this absolute difference is bigger are selected as feature tokens of the feature vectors, based on which we will classify our corpus files. Using the feature tokens of this formula, slightly worse accuracy has been observed. In the end of the program "FeatureSelectionWithIG.py", a dictionary will be created, which contains the feature tokens that will be used for the Naive-Bayes classifier and it is saved in a file called *"feature_dictionary.txt"*.

## Perform Naive-Bayes classification

After the feature selection step, run *"MyCustomNaiveBayesSpamHam.py"* to start the classification process. Run:
```python
python MyCustomNaiveBayesSpamHam.py
```
The classifier checks for each test document if its feature vector is exactly the same as any train document feature vector. If that is the case, the specified test document is classified in the same category of that train document. In case the feature vector of the test document does not match the feature vector of any train document, we use Laplace probability estimates to determine its correct category. The Laplace probability estimate classification calculates the conditional probability of the test document feature vector for each 2 categories (spam or ham) and categorizes the document in the class for which the conditional probability is higher. I.e. for the feature vector x<sub>i</sub> of the test document i we calculate the probabilities p(x<sub>i</sub>|C=1) and p(x<sub>i</sub>|C=0) (reminder: C=1 for spam, C=0 for ham) and the class that has higher probability will be the predicted class of the train document. For the calculation of the probabilies we also use Laplace smoothing, adding 1 in the enumerator and 2 (for 2 classes spam and ham) in the denominator. This is the formula that is used for calculating the probability of the feature token i belonging to a spam document:

![Laplace Smoothing token](http://latex.codecogs.com/gif.latex?%5Cfrac%7BspamDocumentFrequencyOfToken%5Bi%5D%20&plus;%201%7D%20%7BnumberOfSpamDocuments%20&plus;%20numberOfClasses%7D%20%3D%20%5Cfrac%7BspamDocumentFrequencyOfToken%5Bi%5D%20&plus;%201%7D%20%7BnumberOfSpamDocuments%20&plus;%202%7D)

To calculate the probability of the entire feature vector belonging to the spam class we multiply the probability of each separate feature token belonging to the spam class. The exact formula is:

![Laplace Smoothing vector](http://latex.codecogs.com/gif.latex?probOfFeatureVectorBelongingToSpam%20%3D%20%5Cfrac%7BP%28C%3D1%29%7D%7BP%28featureVector%29%7D%20%5Ccdot%20%5Cprod_i%20%5Cfrac%7BspamDocumentFrequencyOfToken%5Bi%5D%20&plus;%201%7D%20%7BnumberOfSpamDocuments%20&plus;%202%7D)

We do the same for the ham class. We can omit the term *P(featureVector)* since it's the same for both the 2 classes. The bigger probability of the two defines the category that the given document and its feature vector is more likely to belong to.


The execution results of the classifier delivered an accuracy of 92.07% while using 100 features tokens and about 94.03% while using 1000 feature tokens. The classifier should have a higher accuracy on a much smaller corpus. For example, in a corpus with about 1000 train texts and 100 test texts that uses a good amount of features, around 1000, the accuracy would be above 95%.
Feel free to make modifications in the code.

**Notes**
<ol>
<li>You can use your own Train and Test text files if you want, as long as they contain "spam" or "ham" in their names, according to their category. The existence of the substring "spam" or "ham" in a text file defines in which category of the two the text file belongs to.</li>
<li>Console execution results are included in the "console_outputs" folder. One output is contained in the file "naive_bayes_no_of_features_1000.txt" where 1000 feature tokens were used and another output is contained in the file "naive_bayes_no_of_features_100.txt" where 100 feature tokens were used.</li>
</ol>
