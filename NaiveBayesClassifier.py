# THIS NAIVE BAYES IMPLEMENTATION IS WRITTEN BY HAND #
# IT USES BOOLEAN FEATURES #

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


def calculate_class_tokens_frequencies(feature_tokens, feature_vector_labels, class_label):
    class_tokens_frequencies = dict()  # same size as a feature vector

    # For each feature token count how many documents of the given class contain it.
    for (i, vector) in enumerate(feature_vector_labels):
        if feature_vector_labels[vector] == class_label:
            for j in range(len(feature_tokens)):
                token = feature_tokens[j]
                if vector[j] == 1:
                    if class_tokens_frequencies.__contains__(token):
                        class_tokens_frequencies[token] = class_tokens_frequencies[token] + 1
                    else:
                        class_tokens_frequencies[token] = 1

    return class_tokens_frequencies


def calculate_laplace_estimate_probability(test_feature_vector, feature_tokens, class_tokens_frequencies, class_frequency, dictionary_size, no_of_train_documents):

    class_probability = class_frequency / no_of_train_documents

    # traditional way: use multiplications of probabilities
    '''
    laplace_estimate_probability = 1
    for (i, token) in enumerate(feature_tokens):
        test_feature = test_feature_vector[i]
        if test_feature == 1:
            if class_tokens_frequencies.__contains__(token):
                probOfTokenBelongingToClass = (class_tokens_frequencies[token] + 1) \
                                              / (class_frequency + dictionary_size)
                laplace_estimate_probability *= probOfTokenBelongingToClass
            else:
                probOfTokenBelongingToClass = (0 + 1) / (class_frequency + dictionary_size)
                laplace_estimate_probability *= probOfTokenBelongingToClass
    laplace_estimate_probability *= class_probability
    '''

    # numerically stable way to avoid multiplications of probabilities
    laplace_estimate_log_probability = 0
    for (i, token) in enumerate(feature_tokens):
        test_feature = test_feature_vector[i]
        if test_feature == 1:
            if class_tokens_frequencies.__contains__(token):
                probOfTokenBelongingToClass = (class_tokens_frequencies[token] + 1) \
                                              / (class_frequency + dictionary_size)
                laplace_estimate_log_probability += math.log(probOfTokenBelongingToClass, 2)
            else:
                probOfTokenBelongingToClass = (0 + 1) / (class_frequency + dictionary_size)
                laplace_estimate_log_probability += math.log(probOfTokenBelongingToClass, 2)
    laplace_estimate_log_probability += math.log(class_probability, 2)

    #return laplace_estimate_probability
    return laplace_estimate_log_probability


###############

# MAIN #

start_time = time.time()

feature_dictionary_dir = "feature_dictionary.txt"

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


# read feature dictionary from file
feature_tokens = read_dictionary_file(feature_dictionary_dir)

print("feature tokens dictionary: ")
print(feature_tokens)
print('')


###############


# training files
print("Reading TRAIN files...")
feature_vector_labels = dict()
for i in range(len(train_files)):
    print('Reading train file ' + "'" + train_files[i] + "'" + '...')

    train_text = ''
    if train_labels[i] == 1:
        train_text = read_file(spam_train_dir + train_files[i])
    elif train_labels[i] == 0:
        train_text = read_file(ham_train_dir + train_files[i])

    train_text_tokens = getTokens(train_text)

    feature_vector = [0] * len(feature_tokens)
    for j in range(len(feature_tokens)):
        if train_text_tokens.__contains__(feature_tokens[j]):
            feature_vector[j] = 1
    feature_vector = tuple(feature_vector)

    if train_labels[i] == 1:  # 1 is for class "SPAM"
        feature_vector_labels[feature_vector] = "SPAM"
    elif train_labels[i] == 0:  # 0 is for class "HAM"
        feature_vector_labels[feature_vector] = "HAM"
print('')


###############


wrong_counter = 0  # the number of wrong classifications made by Logistic Regression
spam_counter = 0  # the number of spam files
ham_counter = 0  # the number of ham files
wrong_spam_counter = 0  # the number of spam files classified as ham
wrong_ham_counter = 0  # the number of ham files classified as spam

spam_class_tokens_frequencies = calculate_class_tokens_frequencies(feature_tokens, feature_vector_labels, "SPAM")
ham_class_tokens_frequencies = calculate_class_tokens_frequencies(feature_tokens, feature_vector_labels, "HAM")

dictionary_size = len(feature_tokens)
no_of_train_documents = len(spam_train_files) + len(ham_train_files)

true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# testing files with Naive Bayes classifier using Laplace estimates
print("Reading TEST files...")
for i in range(len(test_files)):  # for all the test files that exist

    test_text = ''
    if test_true_labels[i] == 1:  # 1 is for class "SPAM"
        test_text = read_file(spam_test_dir + test_files[i])
    if test_true_labels[i] == 0:  # 0 is for class "HAM"
        test_text = read_file(ham_test_dir + test_files[i])
    test_text_tokens = getTokens(test_text)

    feature_vector = [0] * len(feature_tokens)
    for j in range(len(feature_tokens)):
        if test_text_tokens.__contains__(feature_tokens[j]):
            feature_vector[j] = 1
    feature_vector = tuple(feature_vector)

    # Laplace estimate classification #
    spam_laplace_estimate_probability = calculate_laplace_estimate_probability(feature_vector,
                                                                               feature_tokens,
                                                                               spam_class_tokens_frequencies,
                                                                               class_frequency=spam_class_frequency,
                                                                               dictionary_size=dictionary_size,
                                                                               no_of_train_documents=no_of_train_documents)
    #print("spam_laplace_estimate_probability: " + str(spam_laplace_estimate_probability))

    ham_laplace_estimate_probability = calculate_laplace_estimate_probability(feature_vector,
                                                                              feature_tokens,
                                                                              ham_class_tokens_frequencies,
                                                                              class_frequency=ham_class_frequency,
                                                                              dictionary_size=dictionary_size,
                                                                              no_of_train_documents=no_of_train_documents)
    #print("ham_laplace_estimate_probability: " + str(ham_laplace_estimate_probability))

    if spam_laplace_estimate_probability >= ham_laplace_estimate_probability and test_true_labels[i] == 1:
        print("'" + test_files[i] + "'" + " classified as: SPAM -> correct")
        true_positives += 1
    elif spam_laplace_estimate_probability >= ham_laplace_estimate_probability and test_true_labels[i] == 0:
        print("'" + test_files[i] + "'" + " classified as: SPAM -> WRONG!")
        wrong_counter += 1
        false_positives += 1
    elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and test_true_labels[i] == 1:
        print("'" + test_files[i] + "'" + " classified as: HAM -> WRONG!")
        wrong_counter += 1
        false_negatives += 1
    elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and test_true_labels[i] == 0:
        print("'" + test_files[i] + "'" + " classified as: HAM -> correct")
        true_negatives += 1

print('')


###############

# METRICS #

print('Manual Naive-Bayes Classifier: ')
print('number of features used: ' + str(dictionary_size))

print('')

# Accuracy

accuracy = ((len(test_files) - wrong_counter) / len(test_files)) * 100
print("accuracy: " + str(accuracy) + " %")
print('')

# Precision-Recall Report

print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(test_files)) + ' files')
print(true_positives, false_positives, true_negatives, false_negatives)

print('')

spam_precision = true_positives / (true_positives + false_positives) * 100
print("precision for spam files: " + str(spam_precision) + " %")
ham_precision = true_negatives / (true_negatives + false_negatives) * 100
print("precision for ham files: " + str(ham_precision) + " %")

spam_recall = true_positives / (true_positives + false_negatives) * 100
print("recall for spam files: " + str(spam_recall) + " %")
ham_recall = true_negatives / (true_negatives + false_positives) * 100
print("recall for ham files: " + str(ham_recall) + " %")

spam_f1_score = 2 * spam_precision * spam_recall / (spam_precision + spam_recall)
print("f1-score for spam files: " + str(spam_f1_score) + " %")
ham_f1_score = 2 * ham_precision * ham_recall / (ham_precision + ham_recall)
print("f1-score for ham files: " + str(ham_f1_score) + " %")

print('')

###############

print("total duration : %s seconds" % (time.time() - start_time))
