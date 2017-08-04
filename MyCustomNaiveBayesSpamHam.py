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
    text_tokens = set(text_tokens)  # remove duplicate tokens
    
    return text_tokens


def calculate_laplace_estimate_probability(new_feature_vector, feature_tokens, feature_vector_labels, label, label_frequency, no_of_classes, no_of_train_documents):
    laplace_estimate_frequencies = dict()  # same size as a feature vector

    # For each feature token of the new_feature_vector
    # count how many documents of the given class contain it.
    for (i, vector) in enumerate(feature_vector_labels):
        #print('feature vector {0}: {1}, label: {2}'.format(i, vector, feature_vector_labels[vector]))
        if feature_vector_labels[vector] == label:
            for j in range(len(vector)):
                token = feature_tokens[j]
                #print("token: " + token + ", vector[j]: " + str(vector[j]) + ", new_feature_vector[j]: " + str(new_feature_vector[j]))
                if vector[j] == new_feature_vector[j]:
                    if not laplace_estimate_frequencies.__contains__(token):
                        laplace_estimate_frequencies[token] = 1
                    else:
                        laplace_estimate_frequencies[token] = laplace_estimate_frequencies[token] + 1
                else:
                    if not laplace_estimate_frequencies.__contains__(token):
                        laplace_estimate_frequencies[token] = 0

    label_probability = label_frequency / no_of_train_documents
    #print("label_probability: " + str(label_probability))

    # use sum of logs instead of multiplications of probabilities
    laplace_estimate_probability = 1
    for token in feature_tokens:
        if laplace_estimate_frequencies.__contains__(token) and laplace_estimate_frequencies[token] != 0:
            #laplace_estimate_probability *= (laplace_estimate_frequencies[token] + 1) / (label_frequency + no_of_classes)
            laplace_estimate_probability += math.log((laplace_estimate_frequencies[token] + 1) / (label_frequency + no_of_classes))
        else:
            #laplace_estimate_probability *= (0 + 1) / (label_frequency + no_of_classes)
            laplace_estimate_probability += math.log((0 + 1) / (label_frequency + no_of_classes))

    #laplace_estimate_probability *= label_probability
    laplace_estimate_probability += math.log(label_probability)

    return laplace_estimate_probability


###############

# MAIN #

start_time = time.time()

train_dir = "../spam_ham/TRAIN/"
test_dir = "../spam_ham/TEST/"
feature_dictionary_dir = "../spam_ham/feature_dictionary.txt"

train_files = sorted([f for f in listdir(train_dir) if isfile(join(train_dir, f))])
test_files = sorted([f for f in listdir(test_dir) if isfile(join(test_dir, f))])

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


# read feature dictionary from file
feature_tokens = read_dictionary_file(feature_dictionary_dir)

print("feature tokens dictionary:" + "\n")
print(feature_tokens)
print("\n")


###############


# training files
print("training files...")
feature_vector_labels = dict()
for i in range(len(train_files)):
    print('Reading train file ' + "'" + train_files[i] + "'" + '...')

    train_text = read_file(train_dir + train_files[i])

    train_text_tokens = getTokens(train_text)

    feature_vector = [0] * len(feature_tokens)
    for j in range(len(feature_tokens)):
        if train_text_tokens.__contains__(feature_tokens[j]):
            feature_vector[j] = 1
    feature_vector = tuple(feature_vector)

    if train_labels[i] == 1:
        feature_vector_labels[feature_vector] = "SPAM"
    else:
        feature_vector_labels[feature_vector] = "HAM"
    #print(train_files[i] + " is: " + feature_vector_labels[feature_vector])

# print all the feature vectors and their labels
#for (i, vector) in enumerate(feature_vector_labels):
#    print('feature vector {0}: {1}, label: {2}'.format(i, vector, feature_vector_labels[vector]))

print("\n")


###############


wrong_counter = 0  # the number of wrong classifications made by Logistic Regression
spam_counter = 0  # the number of spam files
ham_counter = 0  # the number of ham files
wrong_spam_counter = 0  # the number of spam files classified as ham
wrong_ham_counter = 0  # the number of ham files classified as spam


# testing files with Naive Bayes classifier using Laplace estimates
print("testing files...")
for i in range(len(test_files)):  # for all the test files that exist

    test_text = read_file(test_dir + test_files[i])

    test_text_tokens = getTokens(test_text)

    feature_vector = [0] * len(feature_tokens)
    for j in range(len(feature_tokens)):
        if test_text_tokens.__contains__(feature_tokens[j]):
            feature_vector[j] = 1
    feature_vector = tuple(feature_vector)


    if feature_vector_labels.__contains__(feature_vector):
        if feature_vector_labels[feature_vector] == "SPAM" and "spam" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: " + feature_vector_labels[feature_vector] + " -> correct")
            spam_counter = spam_counter + 1
        elif  feature_vector_labels[feature_vector] == "SPAM" and "ham" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: " + feature_vector_labels[feature_vector] + " -> WRONG!")
            ham_counter = ham_counter + 1
            wrong_ham_counter = wrong_ham_counter + 1
            wrong_counter = wrong_counter + 1
        elif  feature_vector_labels[feature_vector] == "HAM" and "spam" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: " + feature_vector_labels[feature_vector] + " -> WRONG!")
            spam_counter = spam_counter + 1
            wrong_spam_counter = wrong_spam_counter + 1
            wrong_counter = wrong_counter + 1
        elif  feature_vector_labels[feature_vector] == "HAM" and "ham" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: " + feature_vector_labels[feature_vector] + " -> correct")
            ham_counter = ham_counter + 1

    else:
        spam_laplace_estimate_probability = calculate_laplace_estimate_probability(feature_vector,
                                                                           feature_tokens,
                                                                           feature_vector_labels,
                                                                           label="SPAM",
                                                                           label_frequency=spam_label_frequency,
                                                                           no_of_classes=2,
                                                                           no_of_train_documents=len(train_files))
        #print("spam_laplace_estimate_probability: " + str(spam_laplace_estimate_probability))

        ham_laplace_estimate_probability = calculate_laplace_estimate_probability(feature_vector,
                                                                           feature_tokens,
                                                                           feature_vector_labels,
                                                                           label="HAM",
                                                                           label_frequency=ham_label_frequency,
                                                                           no_of_classes=2,
                                                                           no_of_train_documents=len(train_files))
        #print("ham_laplace_estimate_probability: " + str(ham_laplace_estimate_probability))

        if spam_laplace_estimate_probability >= ham_laplace_estimate_probability and "spam" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: SPAM -> correct" + " (laplace estimate classification)")
            spam_counter = spam_counter + 1
        elif spam_laplace_estimate_probability >= ham_laplace_estimate_probability and "ham" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: SPAM -> WRONG!" + " (laplace estimate classification)")
            ham_counter = ham_counter + 1
            wrong_ham_counter = wrong_ham_counter + 1
            wrong_counter = wrong_counter + 1
        elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and "spam" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: HAM -> WRONG!" + " (laplace estimate classification)")
            spam_counter = spam_counter + 1
            wrong_spam_counter = wrong_spam_counter + 1
            wrong_counter = wrong_counter + 1
        elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and "ham" in test_files[i]:
            print("'" + test_files[i] + "'" + " classified as: HAM -> correct" + " (laplace estimate classification)")
            ham_counter = ham_counter + 1

print('\n')


###############

# METRICS #

print('Manual Naive-Bayes Classifier: ')
print('number of features used: ' + str(len(feature_tokens)))

print('\n')

# Accuracy

accuracy = ((len(test_files) - wrong_counter) / len(test_files)) * 100
print("accuracy: " + str(accuracy) + " %")
print("\n")

# Precision-Recall Report

print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(test_files)) + ' files')
print("number of wrong spam classifications: " + str(wrong_spam_counter) + ' out of ' + str(spam_counter) + ' spam files')
print("number of wrong ham classifications: " + str(wrong_ham_counter) + ' out of ' + str(ham_counter) + ' ham files')

print("\n")

spam_precision = (spam_counter - wrong_spam_counter) / (spam_counter - wrong_spam_counter + wrong_ham_counter)
print("precision for spam files: " + str(spam_precision))
ham_precision = (ham_counter - wrong_ham_counter) / (ham_counter - wrong_ham_counter + wrong_spam_counter)
print("precision for ham files: " + str(ham_precision))

spam_recall = (spam_counter - wrong_spam_counter) / (spam_counter)
print("recall for spam files: " + str(spam_recall))
ham_recall = (ham_counter - wrong_ham_counter) / (ham_counter)
print("recall for ham files: " + str(ham_recall))

spam_f1_score = 2 * spam_precision * spam_recall / (spam_precision + spam_recall)
print("f1-score for spam files: " + str(spam_f1_score))
ham_f1_score = 2 * ham_precision * ham_recall / (ham_precision + ham_recall)
print("f1-score for ham files: " + str(ham_f1_score))

print("\n")

###############

print("total duration : %s seconds" % (time.time() - start_time))
