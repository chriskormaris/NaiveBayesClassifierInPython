# THIS NAIVE BAYES IMPLEMENTATION IS WRITTEN BY HAND #
# IT USES BOOLEAN FEATURES #

import math
import time
# I/O Libraries
from os import listdir
from os.path import isfile, join

from nltk import word_tokenize
from nltk.corpus import stopwords

feature_dictionary_dir = "feature_dictionary.txt"

spam_train_dir = "LingspamDataset/spam-train/"
ham_train_dir = "LingspamDataset/nonspam-train/"
spam_test_dir = "LingspamDataset/spam-test/"
ham_test_dir = "LingspamDataset/nonspam-test/"


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


def calculate_token_frequencies_in_class(feature_tokens, stop_words, class_documents):
    token_frequencies_in_class = dict()  # same size as a feature vector
    class_distinct_words = set()
    total_words_in_class = 0

    for token in feature_tokens:
        token_frequencies_in_class[token] = 0

    # For each feature token count how many times the documents of the given class contain it.
    for i, document in enumerate(class_documents):
        # print('document:', document)
        tokenized_document = word_tokenize(document)
        # print('tokenized_document:', tokenized_document)
        filtered_document = [w.lower() for w in tokenized_document if not w.lower() in stop_words]
        # print('filtered_document:', filtered_document)
        for word in filtered_document:
            if word in feature_tokens:
                token_frequencies_in_class[word] = token_frequencies_in_class[word] + 1

        document_set = set(filtered_document)
        class_distinct_words = class_distinct_words.union(document_set)

        # number_of_class_words += len(tokenized_document)
        total_words_in_class += len(filtered_document)

    # number_of_class_words = len(class_distinct_words)
    return token_frequencies_in_class, class_distinct_words, total_words_in_class


def calculate_laplace_estimate_probability(
        test_feature_vector,
        feature_tokens,
        class_probability,
        token_frequencies_in_class,
        total_words_in_class,
        V
):
    # laplace_estimate_probability = 1
    laplace_estimate_log_probability = 0
    for i, test_feature in enumerate(test_feature_vector):
        token = feature_tokens[i]
        if test_feature == 1:
            if token_frequencies_in_class.__contains__(token):
                probOfTokenBelongingToClass = (token_frequencies_in_class[token] + 1) / (total_words_in_class + V)
            else:
                probOfTokenBelongingToClass = (0 + 1) / (total_words_in_class + V)

            # traditional way: using multiplications of probabilities
            # laplace_estimate_probability *= probOfTokenBelongingToClass

            # numerically stable way to avoid multiplications of probabilities
            laplace_estimate_log_probability += math.log(probOfTokenBelongingToClass, 2)

    # laplace_estimate_probability *= class_probability
    laplace_estimate_log_probability += math.log(class_probability, 2)

    # return laplace_estimate_probability
    return laplace_estimate_log_probability


###############

# MAIN #

if __name__ == '__main__':

    start_time = time.time()

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

    # print("feature tokens dictionary: ")
    # print(feature_tokens)
    # print('')

    stop_words = set(stopwords.words('english'))

    ###############

    # training files
    print("Reading TRAIN files...")
    spam_train_documents = []
    ham_train_documents = []
    for i in range(len(train_files)):
        if train_labels[i] == 1:  # for "SPAM" files
            spam_train_document = read_file(spam_train_dir + train_files[i])
            # candidate_features = getTokens(train_text)
            spam_train_documents.append(spam_train_document)
        elif train_labels[i] == 0:  # for "HAM" files
            ham_train_document = read_file(ham_train_dir + train_files[i])
            # candidate_features = getTokens(train_text)
            ham_train_documents.append(ham_train_document)

    print('DONE\n')

    ###############

    print("Calculating feature token frequencies in SPAM files...")
    token_frequencies_in_spam_class, spam_distinct_words, total_words_in_spam_class = \
        calculate_token_frequencies_in_class(feature_tokens, stop_words, spam_train_documents)
    print('DONE\n')

    print("Calculating feature token frequencies in HAM files...")
    token_frequencies_in_ham_class, ham_distinct_words, total_words_in_ham_class = \
        calculate_token_frequencies_in_class(feature_tokens, stop_words, ham_train_documents)
    print('DONE\n')

    # FOR DEBUGGING
    # print('token frequencies in spam class:')
    # print(sorted(token_frequencies_in_spam_class.items()))
    # print('token frequencies in ham class:')
    # print(sorted(token_frequencies_in_ham_class.items()))
    # print('')

    # print(spam_distinct_words)
    # print(ham_distinct_words)
    # print('')

    V = len(spam_distinct_words.union(ham_distinct_words))

    print('total words in spam class:', total_words_in_spam_class)
    print('total words in ham class:', total_words_in_ham_class)
    print('vocabulary size |V|:', V)
    print('')

    wrong_counter = 0  # the number of wrong classifications made by Logistic Regression

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
        test_text_tokens = word_tokenize(test_text)
        filtered_test_text_tokens = [w.lower() for w in test_text_tokens if not w.lower() in stop_words]

        test_feature_vector = [0] * len(feature_tokens)
        for j in range(len(feature_tokens)):
            if test_text_tokens.__contains__(feature_tokens[j]):
                test_feature_vector[j] = 1

        # Laplace estimate classification #
        spam_laplace_estimate_probability = calculate_laplace_estimate_probability(
            test_feature_vector,
            feature_tokens,
            spam_class_probability,
            token_frequencies_in_spam_class,
            total_words_in_spam_class,
            V
        )
        # print("spam_laplace_estimate_probability: " + str(spam_laplace_estimate_probability))

        ham_laplace_estimate_probability = calculate_laplace_estimate_probability(
            test_feature_vector,
            feature_tokens,
            ham_class_probability,
            token_frequencies_in_ham_class,
            total_words_in_ham_class,
            V
        )
        # print("ham_laplace_estimate_probability: " + str(ham_laplace_estimate_probability))

        if spam_laplace_estimate_probability >= ham_laplace_estimate_probability and test_true_labels[i] == 1:
            print("'" + test_files[i] + "'" + " classified as: SPAM -> correct")
            true_positives += 1
        elif spam_laplace_estimate_probability >= ham_laplace_estimate_probability and test_true_labels[i] == 0:
            print("'" + test_files[i] + "'" + " classified as: SPAM -> WRONG!")
            wrong_counter += 1
            false_positives += 1
        elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and test_true_labels[i] == 0:
            print("'" + test_files[i] + "'" + " classified as: HAM -> correct")
            true_negatives += 1
        elif spam_laplace_estimate_probability < ham_laplace_estimate_probability and test_true_labels[i] == 1:
            print("'" + test_files[i] + "'" + " classified as: HAM -> WRONG!")
            wrong_counter += 1
            false_negatives += 1

    print('')

    ###############

    # METRICS #

    print('Manual Naive-Bayes Classifier: ')
    print('number of features used: ' + str(len(feature_tokens)))

    print('')

    # Accuracy

    accuracy = ((len(test_files) - wrong_counter) / len(test_files)) * 100
    print("accuracy: " + str(accuracy) + " %")
    print('')

    # Precision-Recall Report

    print("number of wrong classifications: " + str(wrong_counter) + ' out of ' + str(len(test_files)) + ' files')
    print(
        "number of wrong spam classifications: " + str(false_positives) + ' out of ' + str(len(test_files)) + ' files')
    print("number of wrong ham classifications: " + str(false_negatives) + ' out of ' + str(len(test_files)) + ' files')

    # print(true_positives, false_positives, true_negatives, false_negatives)

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
