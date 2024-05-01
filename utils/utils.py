import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold


class Utils:
    """
    A class containing helper functions used throughout the project.
    """
    @staticmethod
    def balance_dataset(df, target_column='TARGET', random_state=42):
        """
        Balance the dataset by upsampling the minority class to match the number of instances in the majority class.

        Parameters:
        - df: DataFrame, the dataset to be balanced
        - target_column: str, the name of the target variable column
        - random_state: int, seed for random number generator

        Returns:
        - balanced_df: DataFrame, the balanced dataset
        """

        majority_class = df[df[target_column] == 0]
        minority_class = df[df[target_column] == 1]

        minority_upsampled = resample(minority_class,
                                      replace=True,
                                      n_samples=len(majority_class),
                                      random_state=random_state)

        balanced_df = pd.concat([majority_class, minority_upsampled])

        return balanced_df

    @staticmethod
    def train_test_split_data(X, y, test_size=0.2, random_state=42):
        """
        Perform train-test split on the dataset.

        Parameters:
        - X: DataFrame, features of the dataset
        - y: Series, target variable of the dataset
        - test_size: float, the proportion of the dataset to include in the test split
        - random_state: int, seed for random number generator

        Returns:
        - X_train, X_test, y_train, y_test: DataFrames/Series, the split features and target variables
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_models(models, X_train, y_train):
        """
        Train multiple models using k-fold cross-validation.

        Parameters:
        - models: dict, a dictionary containing model names as keys and model instances as values
        - X_train: DataFrame, features of the training dataset
        - y_train: Series, target variable of the training dataset

        Returns:
        - trained_models: dict, trained models
        """

        # Set up cross-validation
        k_folds = 5
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Initialize trained models dictionary
        trained_models = {}

        # Train models
        for name, model in models.items():
            print(f"Training {name}...")
            trained_models[name] = []
            for train_index, val_index in skf.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
                model.fit(X_train_fold, y_train_fold)
                trained_models[name].append(model)

        return trained_models

    @staticmethod
    def evaluate_models(trained_models, X_test, y_test):
        """
        Evaluate multiple models on the test dataset.

        Parameters:
        - trained_models: dict, trained models
        - X_test: DataFrame, features of the test dataset
        - y_test: Series, target variable of the test dataset

        Returns:
        - evaluation_results: dict, evaluation metrics for each model on the test dataset
        """

        # Initialize evaluation results dictionary
        evaluation_results = {}

        # Evaluate models
        for name, model_list in trained_models.items():
            print(f"Evaluating {name}...")
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            for model in model_list:
                y_pred = model.predict(X_test)
                accuracy_scores.append(accuracy_score(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred))
                recall_scores.append(recall_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred))
            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            avg_precision = sum(precision_scores) / len(precision_scores)
            avg_recall = sum(recall_scores) / len(recall_scores)
            avg_f1_score = sum(f1_scores) / len(f1_scores)
            evaluation_results[name] = {
                "Average Accuracy": avg_accuracy,
                "Average Precision": avg_precision,
                "Average Recall": avg_recall,
                "Average F1 Score": avg_f1_score
            }
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Average F1 Score: {avg_f1_score:.4f}")
            print()

        return evaluation_results
