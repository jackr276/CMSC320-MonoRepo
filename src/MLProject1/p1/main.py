import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

import datasets
import utils
import simple_classifier
import knn
import perceptron
import dt


def analysis_part1():
    print("Part 1: Loading and Processing Data...")
    # Fetch raw data
    print("Fetching MNIST data (this might take a few seconds)...")
    X_raw, y_raw = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    
    # Process the data using your implemented function in utils
    X_train, X_test, y_train, y_test = utils.process_data(X_raw, y_raw)
    
    # Q1: Plot a 3 and an 8
    # TODO: Find one example of a +1 (an '8') and one example of a -1 (a '3') in X_train.
    # Use utils.plot_images() to display them side-by-side.
    
    # Q2: Print the shapes of the training and testing sets.
    # TODO: Print the shape of X_train, y_train, X_test, and y_test.
    
    # Q3: Print the number of positive (+1) and negative (-1) examples in both sets.
    # TODO: Count and print how many +1s and -1s are in y_train and y_test.
    
    # Q4: Most Frequent Baseline Evaluation
    clf = simple_classifier.MostFrequentClassClassifier()
    clf.train(X_train, y_train)
    preds = clf.predict(X_test)
    
    # TODO: Compute and print the test accuracy of the MostFrequentClassClassifier.
    # Hint: Use utils.compute_accuracy(y_test, preds)

def analysis_part2():
    print("\nPart 2 (KNN on 10% of CIFAR-10):")
    
    # Fetch CIFAR-10 data
    print("Fetching CIFAR-10 dataset...")
    X_raw_cifar, y_raw_cifar = utils.fetch_cifar10()
    
    # Process the data
    X_train_c, X_test_c, y_train_c, y_test_c = utils.process_cifar_data(X_raw_cifar, y_raw_cifar)
    
    # Q5: Visualizing Neighbors for a Correct Prediction
    # TODO: Train KNN(k=5). 
    # Find a test example where the prediction is CORRECT.
    # Get its 5 nearest neighbors from the training set.
    # Plot using utils.plot_image_and_neighbors.
    
    # Q6: Visualizing Neighbors for a Mistake
    # TODO: Find a test example where the prediction is WRONG.
    # Get its 5 nearest neighbors from the training set.
    # Plot using utils.plot_image_and_neighbors.
    
    # Q7: Hyperparameters, Overfitting, and Underfitting
    k_vals = [3, 5, 7, 9, 11, 13]
    train_accs = ...
    test_accs = ...
    # TODO: Loop over k, train the model, get Train Acc and Test Acc.
    # Plot Train and Test accuracies vs. k.
    # Hint: plotting code below
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, train_accs, marker='o', label='Train Accuracy')
    plt.plot(k_vals, test_accs, marker='s', label='Test Accuracy')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs. K (Airplane vs Frog)')
    plt.legend()
    plt.grid(True)
    plt.show()

def analysis_part3():
    print("\nPart 3 (Perceptron):")
    
    # Q9 & Q10: Blob Dataset Analysis
    print("Fetching blob data...")
    X_train_blob, X_test_blob, y_train_blob, y_test_blob = utils.get_blob_data()
    
    # TODO: Train a Perceptron for 50 epochs on the blob training data.
    p_blob = perceptron.Perceptron(num_epochs=50)
    p_blob.train(X_train_blob, y_train_blob)

    # TODO: Compute and print the final training and testing accuracies.
    y_train_pred_blob = p_blob.predict(X_train_blob)
    y_test_pred_blob = p_blob.predict(X_test_blob)

    train_acc_blob = utils.compute_accuracy(y_train_blob, y_train_pred_blob)
    test_acc_blob = utils.compute_accuracy(y_test_blob, y_test_pred_blob)

    print("blob training accuracy:", train_acc_blob)
    print("blob testing accuracy:", test_acc_blob)

    # TODO: Use utils.plot_decision_boundary to visualize the model on the blob data.
    utils.plot_decision_boundary(X_train_blob, y_train_blob, p_blob)
    
    # Q11: Collinear Blobs Problem
    print("\nFetching collinear data...")
    X_collin, y_collin = utils.get_collinear_blobs()

    # TODO: Train a Perceptron for 100 epochs on the collinear data.
    p_collin = perceptron.Perceptron(num_epochs=100)
    p_collin.train(X_collin, y_collin)

    y_collin_pred = p_collin.predict(X_collin)
    train_acc_collin = utils.compute_accuracy(y_collin, y_collin_pred)

    # TODO: Print the final training accuracy.
    print("collinear  training accuracy:", train_acc_collin)

    # TODO: Use utils.plot_decision_boundary to visualize the model on the collinear data.
    utils.plot_decision_boundary(X_collin, y_collin, p_collin)

def analysis_part4():
    print("\n--- Analysis Part 4 (Decision Trees) ---")
    
    # We use datasets.py for explainable features instead of MNIST/CIFAR
    tennis_X, tennis_y = datasets.TennisData.X, datasets.TennisData.Y          # train set
    tennis_Xte, tennis_yte = datasets.TennisData.Xte, datasets.TennisData.Yte  # test set

    sentiment_X, sentiment_y = datasets.SentimentData.X, datasets.SentimentData.Y
    sentiment_Xte, sentiment_yte = datasets.SentimentData.Xte, datasets.SentimentData.Yte
    
    # Q13: Evaluate performance with depths 1, 3, and 5 on SentimentData
    # TODO: Train DT with max_depth 1, 3, and 5 on sentiment_X/y. Evaluate and print accuracy.
    
    # Q14: Learning Curves (Dataset Size)
    # TODO: Generate learning curves by changing the dataset size (e.g., using SentimentData).
    # Hint: use plotting code from above, you may also make it a function and call it from `utils`
    
    # Q15: Hyperparameter Curves (Max Depth)
    # TODO: Generate hyperparameter curves by varying the max_depth hyperparameter on SentimentData.

if __name__ == "__main__":
    # You can comment/uncomment these out to run specific parts
    #analysis_part1()
    # analysis_part2()
     analysis_part3()
    # analysis_part4()
