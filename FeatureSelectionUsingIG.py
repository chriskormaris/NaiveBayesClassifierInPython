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
    text_tokens = re.findall(r"[\w']+", train_text)
    # remove digits, special characters and convert to lowercase
    for k in range(len(text_tokens)):
        text_tokens[k] = text_tokens[k].lower()
        text_tokens[k] = text_tokens[k].replace("_", "")
        text_tokens[k] = re.sub("[0-9]+", "", text_tokens[k])
    text_tokens = set(text_tokens)  # convert list to set, in order to remove duplicate tokens

    return text_tokens


def getStopwords(stopwords_file):
    stopwords = []
    # Load stopwords
    with open(stopwords_file, 'r') as f:
        for line in f:
            stopwords.append(line.split()[0])

    return stopwords


def write_tokens_to_file(tokens, filename):
    f = open(filename, 'w')
    for token in tokens:
        f.write(token + '\n')
    f.close()


###############

# MAIN #

train_dir = "../part02SpamHam/TRAIN/"
ham_feature_dictionary_dir = "ham_feature_dictionary.txt"
spam_feature_dictionary_dir = "spam_feature_dictionary.txt"
stopwords_dir = "stopwords.txt"

train_files = sorted([f for f in listdir(train_dir) if isfile(join(train_dir, f))])
train_labels = read_labels(train_files)
stopwords = getStopwords(stopwords_dir)

spam_label_frequency = get_label_frequency(train_labels, 1)  # 1 is for SPAM, 0 is for HAM
print("number of SPAM train documents: " + str(spam_label_frequency))
ham_label_frequency = get_label_frequency(train_labels, 0)  # 1 is for SPAM, 0 is for HAM
print("number of HAM train documents: " + str(ham_label_frequency))

spam_label_probability = spam_label_frequency / len(train_files)
print("SPAM train document probability: " + str(spam_label_probability))
ham_label_probability = ham_label_frequency / len(train_files)
print("HAM train document probability: " + str(ham_label_probability))

print()

###############

# feature selection with Information Gain #

# number of features
no_spam_features = 500
no_ham_features = 500

# a dictionary which has as a key how many documents a feature (token) appears in
feature_frequency = dict()

# a dictionary which has as a key how many train spam documents a feature appears in
feature_spam_frequency = dict()

# a dictionary which has as a key how many train ham documents a feature appears in
feature_ham_frequency = dict()

# calculate feature_frequencies dict
for i in range(len(train_files)):
    train_text = read_file(train_dir + train_files[i])
    candidate_features = getTokens(train_text)

    for token in candidate_features:
        if token not in stopwords:
            if not feature_frequency.__contains__(token):
                feature_frequency[token] = 1
            else:
                feature_frequency[token] = feature_frequency[token] + 1

            if train_labels[i] == 1:
                if not feature_spam_frequency.__contains__(token):
                    feature_spam_frequency[token] = 1
                    feature_ham_frequency[token] = 0
                else:
                    feature_spam_frequency[token] = feature_spam_frequency[token] + 1
            elif train_labels[i] == 0:
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
spam_IG = dict()
ham_IG = dict()

# First calculate the entropy of the dataset
H_C = - (spam_label_probability * math.log(spam_label_probability) + ham_label_probability * math.log(ham_label_probability))

print('entropy of the dataset: H(C) = ' + str(H_C))

# precaution to avoid division by zero and log(0)
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
        H_C_given_X1 = - (P_C1_given_X1 * math.log(P_C1_given_X1 + error) + P_C0_given_X1 * math.log(P_C0_given_X1 + error))

        # bayes rule: P(C=1|Xi=0) = P(Xi=0|C=1) * P(C=1) / P(Xi=0)
        P_C1_given_X0 = (1 - feature_spam_cond_probability[token]) * spam_label_probability / (1 - feature_probability[token] + error)
        # bayes rule: P(C=0|Xi=0) = P(Xi=0|C=0) * P(C=0) / P(Xi=0)
        P_C0_given_X0 = (1 - feature_ham_cond_probability[token]) * ham_label_probability / (1 - feature_probability[token] + error)

        # conditional entropy: H(C|Xi=0)
        H_C_given_X0 = - (P_C1_given_X0 * math.log(P_C1_given_X0 + error) + P_C0_given_X0 * math.log(P_C0_given_X0 + error))

        if feature_spam_cond_probability[token] > feature_ham_cond_probability[token]:
            spam_IG[token] = H_C - (feature_probability[token] * H_C_given_X1 + (1 - feature_probability[token]) * H_C_given_X0)
        else:
            ham_IG[token] = H_C - (feature_probability[token] * H_C_given_X1 + (1 - feature_probability[token]) * H_C_given_X0)

        # IG(C,Xi) = IG(Xi,C) = H(C) - SUM ( P(Xi=x) * H(C|Xi=x) for every x)
        #IG[token] = H_C - (feature_probability[token] * H_C_given_X1 + (1 - feature_probability[token]) * H_C_given_X0)

'''
# ALTERNATIVE IG score calculation implementation
# Calculate the information gain for each candidate feature.
# IG is defined as the difference between the two conditional probabilities.
# The tokens where this difference is higher have higher Information Gain.
feature_ham_probability = dict()
feature_spam_probability = dict()
for (i, token) in enumerate(feature_frequency):
    if token != "":  # exclude the empty string ""
        feature_probability[token] = feature_frequency[token] / len(train_files)
        feature_ham_probability[token] = feature_ham_frequency[token] / ham_label_frequency
        feature_spam_probability[token] = feature_spam_frequency[token] / spam_label_frequency
        
        if feature_ham_probability[token] > feature_spam_probability[token]:
            spam_IG[token] = abs(feature_ham_probability[token] - feature_spam_probability[token])
        else:
            ham_IG[token] = abs(feature_ham_probability[token] - feature_spam_probability[token])

        #IG[token] = abs(feature_ham_probability[token] - feature_spam_probability[token])
'''

# sort IG_ham dictionary in descending order by score (the higher score the better)
ham_IG = OrderedDict(sorted(ham_IG.items(), key=itemgetter(1), reverse=True))

# sort IG_spam dictionary in descending order by score (the higher score the better)
spam_IG = OrderedDict(sorted(spam_IG.items(), key=itemgetter(1), reverse=True))

ham_feature_tokens = []

# create and print the list of the ham feature dictionary tokens and their corresponding ham IG scores
print("ham feature tokens: ")
for (i, token) in enumerate(ham_IG):
    # collect the m ham feature tokens with the highest information gain
    if i < no_ham_features:
        ham_feature_tokens.append(token)
        print(token + ", ham information gain score: " + str(ham_IG[token]))
    else:
        break
print()

#write ham_feature_tokens to file
write_tokens_to_file(ham_feature_tokens, ham_feature_dictionary_dir)

# do the same for class spam

spam_feature_tokens = []

# create and print the list of the spam feature dictionary tokens and their corresponding spam IG scores
print("spam feature tokens: ")
for (i, token) in enumerate(spam_IG):
    # collect the m spam feature tokens with the highest information gain
    if i < no_spam_features:
        spam_feature_tokens.append(token)
        print(token + ", spam information gain score: " + str(spam_IG[token]))
    else:
        break
print()

#write spam_feature_tokens to file
write_tokens_to_file(spam_feature_tokens, spam_feature_dictionary_dir)

print()
