from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def evaluate_preds_classification(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1} \n")
    return metric_dict


def fit_and_score(models: dict, x_train, x_test, y_train, y_test, plot=False) -> dict:
    """
    Fits and evaluates given machine learning models.

    :param models: a dict of different Scikit-Learn machine learning models
    :param x_train: training data (no target values)
    :param x_test: testing data (no target values)
    :param y_train: training target values
    :param y_test: test target values
    :param plot : boolean - If true, also plot will be displayed. False on default.
    :return: dictionary containing scores of the models.
    """

    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    scores = {}
    # Loop through models
    for name, model in models.items():
        model.fit(x_train, y_train)
        # Evaluate the model and append it's score to the scores dict
        y_preds = model.predict(x_test)
        print(f"========== {name} ==========")
        scores[name] = evaluate_preds_classification(y_test, y_preds)
    if plot:
        model_compared = pd.DataFrame(scores.values(), index=models.keys())
        model_compared.plot.bar(figsize=(10, 8))
    return scores


def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().

    :param y_test: True target values
    :param y_preds: Predicted target values
    :return: Matmplotlib's figure and axis
    """
    sns.set(font_scale=1.5)  # Increase font size
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,  # Annotate the boxes
                     cbar=False)
    plt.xlabel("Predicted label")  # predictions go on the x-axis
    plt.ylabel("True label")  # true labels go on the y-axis
    return fig, ax


def plot_correlation_mat(corr_matrix):
    """
    Plots Correlation matrix using Seaborn's heatmap()
    :param corr_matrix: Pandas Correlation matrix
    :return: Figure and axis of the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(corr_matrix, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
    return fig, ax


def cross_validated_report_classification(clf, x, y, plot=False):
    """

    :param clf: Scikit-Learn's classifier
    :param x:   Features
    :param y:   Targets
    :param plot: boolean - If true plot is also being draw. False by default.
    :return:    Pandas DataFrame with containing Cross-validated Accuracy, Precision, Recall, F1 in this order.
    """
    # Cross-validated accuracy
    acc = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="accuracy"))
    print(f"Cross-validated accuracy: {acc}")

    # Cross-validated precision
    precision = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="precision"))
    print(f"Cross-validated precision: {precision}")

    # Cross-validated recall
    rec = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="recall"))
    print(f"Cross-validated recall: {rec:.2f}")

    # Cross-validated F1 score
    f1 = mean_and_round(cross_val_score(clf, x, y, cv=5, scoring="f1"))
    print(f"Cross-validated F1 Score: {f1}")

    metrics = pd.DataFrame({"Accuracy": acc,
                            "Precision": precision,
                            "Recall": rec,
                            "F1 Score": f1}, index=[0])

    if plot:
        metrics.T.plot.bar(title="Cross-Validated model metrics",
                           legend=False, yticks=np.arange(0, 1, 0.1))

    return metrics


def mean_and_round(np_arr):
    """
    Takes an NumPy objects and calculate mean rounded up to 2 decimal places. Uses NumPy.
    :param np_arr: iterable sequence of numbers
    :return: rounded mean of values in iterable
    """

    return np.around(np.mean(np_arr), 2)
