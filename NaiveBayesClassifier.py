# THIS NAIVE BAYES IMPLEMENTATION IS WRITTEN BY HAND #
# IT USES BOOLEAN FEATURES AND SEPARATELY FOR "SPAM" AND "HAM" CLASSES #

# force the result of divisions to be float numbers
from __future__ import division

import re
import time
import math

# I/O Libraries
from os import listdir
from os.path import isfile, join

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


def read_dictionary_file(filename):
    text_file = open(filename, "r")
    lines = text_file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].replace("\n", "")
    return lines


def read_file(filename):
    text_file = open(filename, "r")
    text = text_file.read()
    return text


def get_label_frequency(train_labels, label):
    frequency = 0
    for train_label in train_labels:
        if train_label == label:
            frequency = frequency + 1
    return frequency


# extracts tokens from the given text
def getTokens(text):
    text_tokens = re.findall(r"[\w']+", text)
    # remove digits, special characters and convert to lowercase
    for k in range(len(text_tokens)):
        text_tokens[k] = text_tokens[k].lower()
        text_tokens[k] = text_tokens[k].replace("_", "")
        text_tokens[k] = re.sub("[0-9]+", "", text_tokens[k])
    text_tokens = set(text_tokens)  # convert list to set, in order to remove duplicate tokens
    text_tokens = list(text_tokens)  # convert set back to list

    return text_tokens


def calculate_label_tokens_frequencies(label_feature_tokens, feature_vectors, label):
    laplace_estimate_frequencies = dict()  # same size as a feature vector

    # For each feature token count how many documents of the given class contain it.
    for (i, vector) in enumerate(feature_vectors):
        for j in range(len(label_feature_tokens)):
            token = label_feature_tokens[j]
            if vector[j] >= 1:
                if laplace_estimate_frequencies.__contains__(token):
                    laplace_estimate_frequencies[token] = laplace_estimate_frequencies[token] + vector[j]
                else:
                    laplace_estimate_frequencies[token] = vector[j]

    return laplace_estimate_frequencies


def calculate_laplace_estimate_probability(test_feature_vector, label_feature_tokens, laplace_estimate_frequencies,
                                           label_frequency, no_of_train_documents, dictionary_size):

    label_probability = label_frequency / no_of_train_documents

    # numerically stable way to avoid multiplications of probabilities
    # known as logsumexp trick
    laplace_estimate_exp_probability = 0
    for (i, token) in enumerate(label_feature_tokens):
        test_feature_token_frequency = test_feature_vector[i]
        if test_feature_token_frequency >= 1:
            if laplace_estimate_frequencies.__contains__(token):
                probOfTokenBelongingToLabel = (laplace_estimate_frequencies[token] + 1) \
                                              / (label_frequency + dictionary_size)
                laplace_estimate_exp_probability += math.exp(probOfTokenBelongingToLabel)
            else:
                probOfTokenBelongingToLabel = (0 + 1) / (label_frequency + dictionary_size)
                laplace_estimate_exp_probability += math.exp(probOfTokenBelongingToLabel)
    laplace_estimate_exp_probability += math.exp(label_probability)
    laplace_estimate_log_probability = math.log(laplace_estimate_exp_probability)

    return laplace_estimate_log_probability


###############

# MAIN #

start_time = time.time()

spam_feature_dictionary_dir = "spam_feature_dictionary.txt"
ham_feature_dictionary_dir = "ham_feature_dictionary.txt"

spam_train_dir = "LingspamDataset/spam-train/"
ham_train_dir = "LingspamDataset/nonspam-train/"
spam_test_dir = "LingspamDataset/spam-test/"
ham_test_dir = "LingspamDataset/nonspam-test/"

spam_train_files = sorted([f for f in listdir(spam_train_dir) if isfile(join(spam_train_dir, f))])
ham_train_files = sorted([f for f in listdir(ham_train_dir) if isfile(join(ham_train_dir, f))])
spam_test_files = sorted([f for f in listdir(spam_test_dir) if isfile(join(spam_test_dir, f))])
ham_test_files = sorted([f for f in listdir(ham_test_dir) if isfile(join(ham_test_dir, f))])

train_files = list(spam_train_files)
train_files.extend(ham_train_files)

test_files = list(spam_test_files)
test_files.extend(ham_test_files)

train_labels = [1] * len(spam_train_files)
train_labels.extend([0] * len(ham_train_files))

test_true_labels = [1] * len(spam_test_files)
test_true_labels.extend([0] * len(ham_test_files))

spam_label_frequency = len(spam_train_files)  # 1 is for SPAM, 0 is for HAM
print("number of SPAM train documents: " + str(spam_label_frequency))
ham_label_frequency = len(ham_train_files)  # 1 is for SPAM, 0 is for HAM
print("number of HAM train documents: " + str(ham_label_frequency))

spam_label_probability = spam_label_frequency / (len(spam_train_files) + len(ham_train_files))
print("SPAM train document probability: " + str(spam_label_probability))
ham_label_probability = ham_label_frequency / (len(spam_train_files) + len(ham_train_files))
print("HAM train document probability: " + str(ham_label_probability))

print('')


###############


# read feature dictionary from file
spam_feature_tokens = read_dictionary_file(spam_feature_dictionary_dir)
ham_feature_tokens = read_dictionary_file(ham_feature_dictionary_dir)

print("spam feature tokens dictionary: ")
print(spam_feature_tokens)
print('')
print("ham feature tokens dictionary: ")
print(ham_feature_tokens)
print('')


###############


# training files
print("Reading TRAIN files...")
spam_feature_vectors = []
ham_feature_vectors = []
for i in range(len(train_files)):
    print('Reading train file ' + "'" + train_files[i] + "'" + '...')

    train_text = ''
    if train_labels[i] == 1:
        train_text = read_file(spam_train_dir + train_files[i])
    elif train_labels[i] == 0:
        train_text = read_file(ham_train_dir + train_files[i])

    train_text_tokens = getTokens(train_text)

    if train_labels[i] == 1:  # 1 is for class "SPAM"
        spam_feature_vector = [0] * len(spam_feature_tokens)
        for j in range(len(spam_feature_tokens)):
            if train_text_tokens.__contains__(spam_feature_tokens[j]):
                spam_feature_vector[j] = 1
        spam_feature_vector = tuple(spam_feature_vector)
        spam_feature_vectors.append(spam_feature_vector)
    if train_labels[i] == 0:  # 0 is for class "HAM"
        ham_feature_vector = [0] * len(ham_feature_tokens)
        for j in range(len(ham_feature_tokens)):
            if train_text_tokens.__contains__(ham_feature_tokens[j]):
                ham_feature_vector[j] = 1
        ham_feature_vector = tuple(ham_feature_vector)
        ham_feature_vectors.append(ham_feature_vector)

print('')


###############


wrong_counter = 0  # the number of wrong classifications made by Logistic Regression
spam_counter = 0  # the number of spam files
ham_counter = 0  # the number of ham files
wrong_spam_counter = 0  # the number of spam files classified as ham
wrong_ham_counter = 0  # the number of ham files classified as spam

spam_feature_tokens_frequencies = calculate_label_tokens_frequencies(spam_feature_tokens, spam_feature_vectors, "SPAM")
ham_feature_tokens_frequencies = calculate_label_tokens_frequencies(ham_feature_tokens, ham_feature_vectors, "HAM")

spam_dictionary_size = len(spam_feature_tokens)
ham_dictionary_size = len(ham_feature_tokens)
no_of_train_documents = len(train_files)

# testing files with Naive Bayes classifier using Laplace estimates
print("Reading TEST files...")
for i in range(len(test_files)):  # for all the test files that exist

    test_text = ''
    if test_true_labels[i] == 1:  # 1 is for class "SPAM"
        test_text = read_file(spam_test_dir + test_files[i])
    elif test_true_labels[i] == 0:  # 0 is for class "HAM"
        test_text = read_file(ham_test_dir + test_files[i])

    test_text_tokens = getTokens(test_text)

    test_spam_feature_vector = [0] * len(spam_feature_tokens)
    for j in range(len(spam_feature_tokens)):
        if test_text_tokens.__contains__(spam_feature_tokens[j]):
            test_spam_feature_vector[j] = 1
    test_spam_feature_vector = tuple(test_spam_feature_vector)

    test_ham_feature_vector = [0] * len(ham_feature_tokens)
    for j in range(len(ham_feature_tokens)):
        if test_text_tokens.__contains__(ham_feature_tokens[j]):
            test_ham_feature_vector[j] = 1
    test_ham_feature_vector = tuple(test_ham_feature_vector)

    # Laplace estimate classification #
    spam_laplace_estimate_probability = calculate_laplace_estimate_probability(test_spam_feature_vector,
                                                                      spam_feature_tokens,
                                                                      spam_feature_tokens_frequencies,
                                                                      label_frequency=spam_label_frequency,
                                                                      no_of_train_documents=no_of_train_documents,
                                                                      dictionary_size=spam_dictionary_size)

    ham_laplace_estimate_probability = calculate_laplace_estimate_probability(test_ham_feature_vector,
                                                                      ham_feature_tokens,
                                                                      ham_feature_tokens_frequencies,
                                                                      label_frequency=ham_label_frequency,
                                                                      no_of_train_documents=no_of_train_documents,
                                                                      dictionary_size=ham_dictionary_size)

    if spam_laplace_estimate_probability >= ham_laplace_estimate_probability and test_true_labels[i] == 1:
        print("'" + test_files[i] + "'" + " classified as: SPAM -> correct")
        spam_counter = spam_counter + 1
    elif spam_laplace_estimate_probability >= ham_laplace_estimate_probability and test_true_labels[i] == 0:
        print("'" + test_files[i] + "'" + " classified as: SPAM -> WRONG!")
        ham_counter = ham_counter + 1
        wrong_ham_counter = wrong_ham_counter + 1
        wrong_counter = wrong_counter + 1
    elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and test_true_labels[i] == 1:
        print("'" + test_files[i] + "'" + " classified as: HAM -> WRONG!")
        spam_counter = spam_counter + 1
        wrong_spam_counter = wrong_spam_counter + 1
        wrong_counter = wrong_counter + 1
    elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and test_true_labels[i] == 0:
        print("'" + test_files[i] + "'" + " classified as: HAM -> correct")
        ham_counter = ham_counter + 1

print('')


###############

# METRICS #

print('Manual Naive-Bayes Classifier: ')
print('number of spam features used: ' + str(spam_dictionary_size))
print('number of ham features used: ' + str(ham_dictionary_size))

print('')

# Accuracy

accuracy = ((len(test_files) - wrong_counter) / len(test_files)) * 100
print("accuracy: " + str(accuracy) + " %")
print('')

# Precision-Recall Report

print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(test_files)) + ' files')
print("number of wrong spam classifications: " + str(wrong_spam_counter) + ' out of ' + str(spam_counter) + ' spam files')
print("number of wrong ham classifications: " + str(wrong_ham_counter) + ' out of ' + str(ham_counter) + ' ham files')

print('')

spam_precision = (spam_counter - wrong_spam_counter) / (spam_counter - wrong_spam_counter + wrong_ham_counter)
print("precision for spam files: " + str(spam_precision))
ham_precision = (ham_counter - wrong_ham_counter) / (ham_counter - wrong_ham_counter + wrong_spam_counter)
print("precision for ham files: " + str(ham_precision))

spam_recall = (spam_counter - wrong_spam_counter) / spam_counter
print("recall for spam files: " + str(spam_recall))
ham_recall = (ham_counter - wrong_ham_counter) / ham_counter
print("recall for ham files: " + str(ham_recall))

spam_f1_score = 2 * spam_precision * spam_recall / (spam_precision + spam_recall)
print("f1-score for spam files: " + str(spam_f1_score))
ham_f1_score = 2 * ham_precision * ham_recall / (ham_precision + ham_recall)
print("f1-score for ham files: " + str(ham_f1_score))

print('')

###############

print("total duration : %s seconds" % (time.time() - start_time))
