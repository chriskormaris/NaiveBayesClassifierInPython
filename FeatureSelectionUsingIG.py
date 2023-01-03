# force the result of divisions to be float numbers
from __future__ import division

from os import listdir
from os.path import isfile, join
import re
import math

# for sorting dictionaries
from collections import OrderedDict
from operator import itemgetter

from nltk.corpus import stopwords

__author__ = 'c.kormaris'


# number of features
m = 1000
# m = 100
# m = 50

spam_train_dir = "LingspamDataset/spam-train/"
ham_train_dir = "LingspamDataset/nonspam-train/"

feature_dictionary_dir = "feature_dictionary.txt"


###############

# FUNCTIONS #

# defines the label of the files based on their names
def read_labels(files):
    labels = []
    for file in files:
        if "spam" in str(file):
            labels.append(1)
        elif "ham" in str(file):
            labels.append(0)
    return labels


def get_label_frequency(train_labels, class_label):
    frequency = 0
    for train_label in train_labels:
        if train_label == class_label:
            frequency = frequency + 1
    return frequency


def read_file(filename):
    text_file = open(filename, "r")
    text = text_file.read()
    return text


# extracts tokens from the given text
def getTokens(text):
    text_tokens = re.findall(r"[\w']+", text)
    # remove digits, special characters and convert to lowercase
    for k in range(len(text_tokens)):
        text_tokens[k] = text_tokens[k].lower()
        text_tokens[k] = text_tokens[k].replace("_", "")
        text_tokens[k] = re.sub("[0-9]+", "", text_tokens[k])
    text_tokens = set(text_tokens)  # convert list to set, in order to remove duplicate tokens

    return text_tokens


def write_tokens_to_file(tokens, filename):
    f = open(filename, 'w')
    for token in tokens:
        f.write(token + '\n')
    f.close()


###############

# MAIN #

spam_train_files = sorted([f for f in listdir(spam_train_dir) if isfile(join(spam_train_dir, f))])
ham_train_files = sorted([f for f in listdir(ham_train_dir) if isfile(join(ham_train_dir, f))])

train_files = list(spam_train_files)
train_files.extend(ham_train_files)

train_labels = [1] * len(spam_train_files)
train_labels.extend([0] * len(ham_train_files))

stop_words = set(stopwords.words('english'))

no_of_train_files = len(train_files)

spam_class_frequency = len(spam_train_files)  # 1 is for SPAM, 0 is for HAM
print("number of SPAM train documents: " + str(spam_class_frequency))
ham_class_frequency = len(ham_train_files)  # 1 is for SPAM, 0 is for HAM
print("number of HAM train documents: " + str(ham_class_frequency))

spam_class_probability = spam_class_frequency / (len(spam_train_files) + len(ham_train_files))
print("SPAM train document probability: " + str(spam_class_probability))
ham_class_probability = ham_class_frequency / (len(spam_train_files) + len(ham_train_files))
print("HAM train document probability: " + str(ham_class_probability))

print('')

###############

# feature selection with Information Gain #

# a dictionary which has as a key how many documents a feature (token) appears in
feature_frequency = dict()

# a dictionary which has as a key how many train spam documents a feature appears in
feature_spam_frequency = dict()

# a dictionary which has as a key how many train ham documents a feature appears in
feature_ham_frequency = dict()

print('Calculating the frequency of each token...')

# calculate feature_frequencies dict
for i in range(len(train_files)):
    train_text = ''
    if train_labels[i] == 1:  # for "SPAM" files
        train_text = read_file(spam_train_dir + train_files[i])
    elif train_labels[i] == 0:  # for "HAM" files
        train_text = read_file(ham_train_dir + train_files[i])
    candidate_features = getTokens(train_text)

    for token in candidate_features:
        if token not in stop_words:
            if not feature_frequency.__contains__(token):
                feature_frequency[token] = 1
            else:
                feature_frequency[token] = feature_frequency[token] + 1

            if train_labels[i] == 1:  # for "SPAM" files
                if not feature_spam_frequency.__contains__(token):
                    feature_spam_frequency[token] = 1
                    feature_ham_frequency[token] = 0
                else:
                    feature_spam_frequency[token] = feature_spam_frequency[token] + 1
            elif train_labels[i] == 0:  # for "HAM" files
                if not feature_ham_frequency.__contains__(token):
                    feature_ham_frequency[token] = 1
                    feature_spam_frequency[token] = 0
                else:
                    feature_ham_frequency[token] = feature_ham_frequency[token] + 1

# sort feature_frequency dictionary in descending order by frequency
feature_frequency = OrderedDict(sorted(feature_frequency.items(), key=itemgetter(1), reverse=True))

###############

# a dictionary which has as a key the probability of a feature appearing in a document
feature_probability = dict()

# a dictionary which has as a key the probability of a feature appearing in a train spam document
feature_spam_cond_probability = dict()

# a dictionary which has as a key the probability of a feature appearing in a train ham document
feature_ham_cond_probability = dict()

# a dictionary that contains the Information gain for each token
IG = dict()

# First calculate the entropy of the dataset
H_C = - (spam_class_probability * math.log(spam_class_probability) + ham_class_probability * math.log(ham_class_probability))

print('entropy of the dataset: H(C) = ' + str(H_C))

# precaution to avoid division by zero and log(0)
error = 1e-7

# Calculate the information gain for each candidate feature.
# The feature that decreases the entropy less is the most desired feature,
# because that means that it is capable of achieving better classification.
for (i, token) in enumerate(feature_frequency):
    if token != "":  # exclude the empty string ""
        feature_probability[token] = feature_frequency[token] / no_of_train_files  # P(Xi=1)
        feature_ham_cond_probability[token] = feature_ham_frequency[token] / ham_class_frequency  # P(Xi=1|C=0)
        feature_spam_cond_probability[token] = feature_spam_frequency[token] / spam_class_frequency  # P(Xi=1|C=1)

        # bayes rule: P(C=1|Xi=1) = P(Xi=1|C=1) * P(C=1) / P(Xi=1)
        P_C1_given_X1 = feature_spam_cond_probability[token] * spam_class_probability / (feature_probability[token] + error)
        # bayes rule: P(C=0|Xi=1) = P(Xi=1|C=0) * P(C=0) / P(Xi=1)
        P_C0_given_X1 = feature_ham_cond_probability[token] * ham_class_probability / (feature_probability[token] + error)

        # conditional entropy: H(C|Xi=1)
        H_C_given_X1 = - (P_C1_given_X1 * math.log(P_C1_given_X1 + error) + P_C0_given_X1 * math.log(P_C0_given_X1 + error))

        # bayes rule: P(C=1|Xi=0) = P(Xi=0|C=1) * P(C=1) / P(Xi=0)
        P_C1_given_X0 = (1 - feature_spam_cond_probability[token]) * spam_class_probability / (1 - feature_probability[token] + error)
        # bayes rule: P(C=0|Xi=0) = P(Xi=0|C=0) * P(C=0) / P(Xi=0)
        P_C0_given_X0 = (1 - feature_ham_cond_probability[token]) * ham_class_probability / (1 - feature_probability[token] + error)

        # conditional entropy: H(C|Xi=0)
        H_C_given_X0 = - (P_C1_given_X0 * math.log(P_C1_given_X0 + error) + P_C0_given_X0 * math.log(P_C0_given_X0 + error))

        # IG(C,Xi) = IG(Xi,C) = H(C) - SUM ( P(Xi=X_train) * H(C|Xi=X_train) for every X_train)
        IG[token] = H_C - (feature_probability[token] * H_C_given_X1 + (1 - feature_probability[token]) * H_C_given_X0)

        #print('{0}: P(Xi=1): {1}, P(Xi=1|C=0): {2}, P(Xi=1|C=1): {3}'.format(token, feature_probability[token],
        #                                                                     feature_ham_probability[token],
        #                                                                     feature_spam_probability[token]))

"""
# MY ALTERNATIVE IG score calculation implementation
# Calculate the information gain for each candidate feature.
# IG is defined as the difference between the two conditional probabilities.
# The tokens where this difference is higher have higher Information Gain.
feature_ham_probability = dict()
feature_spam_probability = dict()
for (i, token) in enumerate(feature_frequency):
    if token != "":  # exclude the empty string ""
        feature_probability[token] = feature_frequency[token] / no_of_train_files
        feature_ham_probability[token] = feature_ham_frequency[token] / ham_class_frequency
        feature_spam_probability[token] = feature_spam_frequency[token] / spam_class_frequency

        #IG[token] = feature_probability[token] * abs(feature_ham_probability[token] - feature_spam_probability[token])
        IG[token] = abs(feature_ham_probability[token] - feature_spam_probability[token])

        #print('{0}: P(Xi=1): {1}, P(Xi=1|C=0): {2}, P(Xi=1|C=1): {3}'.format(token, feature_probability[token],
        #                                                                     feature_ham_probability[token],
        #                                                                     feature_spam_probability[token]))
"""


# sort IG dictionary in descending order by score (the higher score the better)
IG = OrderedDict(sorted(IG.items(), key=itemgetter(1), reverse=True))

print('')

feature_tokens = []

# create and print the list of the feature dictionary tokens and their corresponding IG scores
print("feature tokens: ")
for (i, token) in enumerate(IG):
    # collect the m feature tokens with the highest information gain
    if i < m:
        feature_tokens.append(token)
        print(token + ", information gain score: " + str(IG[token]))
    else:
        break
print('')

# write feature_tokens to file
write_tokens_to_file(feature_tokens, feature_dictionary_dir)
