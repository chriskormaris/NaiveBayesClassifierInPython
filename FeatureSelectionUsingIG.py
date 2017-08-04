# force the result of divisions to be float numbers
from __future__ import division

from os import listdir
from os.path import isfile, join
import re
import math

# for sorting dictionaries
from collections import OrderedDict
from operator import itemgetter

__author__ = 'c.kormaris'


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


def get_label_frequency(train_labels, label):
    frequency = 0
    for train_label in train_labels:
        if train_label == label:
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
    text_tokens = set(text_tokens)  # remove duplicate tokens
    
    return text_tokens


def write_tokens_to_file(tokens, filename):
    f = open(filename, 'w')
    for token in tokens:
        f.write(token + '\n')
    f.close()


###############

# MAIN #

train_dir = "TRAIN/"
feature_dictionary_dir = "feature_dictionary.txt"

train_files = sorted([f for f in listdir(train_dir) if isfile(join(train_dir, f))])

train_labels = read_labels(train_files)

spam_label_frequency = get_label_frequency(train_labels, 1)  # 1 is for SPAM, 0 is for HAM
print("number of SPAM train documents: " + str(spam_label_frequency))
ham_label_frequency = get_label_frequency(train_labels, 0)  # 1 is for SPAM, 0 is for HAM
print("number of HAM train documents: " + str(ham_label_frequency))

spam_label_probability = spam_label_frequency / len(train_files)
print("SPAM train document probability: " + str(spam_label_probability))
ham_label_probability = ham_label_frequency / len(train_files)
print("HAM train document probability: " + str(ham_label_probability))

print("\n")


###############


# feature selection with Information Gain #

# number of features
m = 1000
#m = 100

# a dictionary which has as a key how many documents a feature (token) appears in
feature_frequency = dict()

# a dictionary which has as a key how many train spam documents a feature appears in
feature_spam_frequency = dict()

# a dictionary which has as a key how many train ham documents a feature appears in
feature_ham_frequency = dict()

# calculate feature_frequencies dict
for i in range(len(train_files)):
    train_text = read_file(train_dir + train_files[i])
    candidate_features = getTokens(text)

    for (j, token) in enumerate(candidate_features):
        if feature_frequency.__contains__(token) == False:
            feature_frequency[token] = 1
        else:
            feature_frequency[token] = feature_frequency[token] + 1

        if train_labels[i] == 1:
            if feature_spam_frequency.__contains__(token) == False:
                feature_spam_frequency[token] = 1
                feature_ham_frequency[token] = 0
            else:
                feature_spam_frequency[token] = feature_spam_frequency[token] + 1
        elif train_labels[i] == 0:
            if feature_ham_frequency.__contains__(token) == False:
                feature_ham_frequency[token] = 1
                feature_spam_frequency[token] = 0
            else:
                feature_ham_frequency[token] = feature_ham_frequency[token] + 1


# sort feature_tokens_dictionary in descending order by frequency
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
H_C = - ( spam_label_probability * math.log(spam_label_probability) + ham_label_probability * math.log(ham_label_probability) )

print('entropy of the dataset: H(C) = ' + str(H_C))

# this is to avoid division by zero and log(0)
error = 1e-7

# Calculate the information gain for each candidate feature.
# The feature that decreases the entropy less is the most desired feature,
# because that means that it is capable of achieving better classification.
for (i, token) in enumerate(feature_frequency):
    if token != "":  # exclude the empty string ""
        feature_probability[token] = feature_frequency[token] / len(train_files)  # P(Xi=1)
        feature_ham_cond_probability[token] = feature_ham_frequency[token] / ham_label_frequency  # P(Xi=1|C=0)
        feature_spam_cond_probability[token] = feature_spam_frequency[token] / spam_label_frequency  # P(Xi=1|C=1)

        # bayes rule: P(C=1|Xi=1) = P(Xi=1|C=1) * P(C=1) / P(Xi=1)
        P_C1_given_X1 = feature_spam_cond_probability[token] * spam_label_probability / (feature_probability[token] + error)
        # bayes rule: P(C=0|Xi=1) = P(Xi=1|C=0) * P(C=0) / P(Xi=1)
        P_C0_given_X1 = feature_ham_cond_probability[token] * ham_label_probability / (feature_probability[token] + error)

        # conditional entropy: H(C|Xi=1)
        H_C_given_X1 = - ( P_C1_given_X1 * math.log(P_C1_given_X1 + error) + P_C0_given_X1 * math.log(P_C0_given_X1 + error))

        # bayes rule: P(C=1|Xi=0) = P(Xi=0|C=1) * P(C=1) / P(Xi=0)
        P_C1_given_X0 = (1 - feature_spam_cond_probability[token]) * spam_label_probability / (1 - feature_probability[token] + error)
        # bayes rule: P(C=0|Xi=0) = P(Xi=0|C=0) * P(C=0) / P(Xi=0)
        P_C0_given_X0 = (1 - feature_ham_cond_probability[token]) * ham_label_probability / (1 - feature_probability[token] + error)

        # conditional entropy: H(C|Xi=0)
        H_C_given_X0 = - ( P_C1_given_X0 * math.log(P_C1_given_X0 + error) + P_C0_given_X0 * math.log(P_C0_given_X0 + error) )

        # IG(C,Xi) = IG(Xi,C) = H(C) - SUM ( P(Xi=x) * H(C|Xi=x) for every x)
        IG[token] = H_C - ( feature_probability[token] * H_C_given_X1 + (1 - feature_probability[token]) * H_C_given_X0 )

        #print('{0}: P(Xi=1): {1}, P(Xi=1|C=0): {2}, P(Xi=1|C=1): {3}'.format(token, feature_probabilities[token],
        #                                                                     feature_ham_probabilities[token],
        #                                                                     feature_spam_probabilities[token]))


'''
# ALTERNATIVE IG score calculation implementation
# Calculate the information gain for each candidate feature.
# IG is defined as the difference between the two conditional probabilities.
# The tokens where this difference is higher have higher Information Gain.
for (i, token) in enumerate(feature_frequencies):
    if token != "":  # exclude the empty string ""
        feature_probabilities[token] = feature_frequencies[token] / len(train_files)
        feature_ham_probabilities[token] = feature_ham_frequencies[token] / ham_label_frequency
        feature_spam_probabilities[token] = feature_spam_frequencies[token] / spam_label_frequency

        #IG[token] = feature_probabilities[token] * abs(feature_ham_probabilities[token] - feature_spam_probabilities[token])
        IG[token] = abs(feature_ham_probabilities[token] - feature_spam_probabilities[token])

        #print('{0}: P(Xi=1): {1}, P(Xi=1|C=0): {2}, P(Xi=1|C=1): {3}'.format(token, feature_probabilities[token],
        #                                                                     feature_ham_probabilities[token],
        #                                                                     feature_spam_probabilities[token]))
'''


# sort IG dictionary in descending order by score (the higher score the better)
IG = OrderedDict(sorted(IG.items(), key=itemgetter(1), reverse=True))

print("\n")

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
print("\n")

#write spam_test tokens to file
write_tokens_to_file(feature_tokens, feature_dictionary_dir)
