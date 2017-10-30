from abc import ABCMeta, abstractmethod
from enum import Enum

import graphlab
import h2o
import pandas as pd
from h2o.estimators import H2ONaiveBayesEstimator
from h2o.estimators import H2ORandomForestEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import naive_bayes
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split

from utils import DatasetFileManager

K_FOLD = 10
SPLIT_RATIO = 0.3
SEED = 2
COLUMNS_INDEX = 1
DATASET_1 = "numerai_training_data"
DATASET_2 = "Skin_NonSkin"


class Framework(metaclass=ABCMeta):
    @classmethod
    def run(cls):
        print("Running Framework {0}...".format(cls.__name__))
        cls.init()
        cls.run_encrypted_stock_market_dataset()
        cls.run_skin_dataset()

    @classmethod
    @abstractmethod
    def init(cls):
        pass

    @staticmethod
    @abstractmethod
    def run_skin_dataset():
        pass

    @staticmethod
    @abstractmethod
    def run_encrypted_stock_market_dataset():
        pass


class GraphLab(Framework):
    class Algorithm_IDENTIFIER(Enum):
        LOGISTIC_REGRESSION = 1,
        RANDOM_FOREST = 2

    @classmethod
    def init(cls):
        pass

    @staticmethod
    def run_skin_dataset():
        print("\n\nRunning Skin dataset...")
        skin_dataset = load_dataset_as_graphlab_data_frame(DATASET_2, "\t")

        # Split 70-30
        train_data, test_data = GraphLab.split_data(skin_dataset)
        GraphLab.run_logistic_regression_split(train_data, test_data)
        GraphLab.run_random_forest_split(train_data, test_data)

        # Cross validation K-fold
        folds = GraphLab.kfold_data(skin_dataset)
        GraphLab.run_logistic_regression_kfold(folds)
        GraphLab.run_random_forest_kfold(folds)

    @staticmethod
    def run_encrypted_stock_market_dataset():
        print("\n\nRunning Encrypted-stock-market-data...")
        encrypted_stock_market_data = load_dataset_as_graphlab_data_frame(DATASET_1, ",")

        # Split 70-30
        train_data, test_data = GraphLab.split_data(encrypted_stock_market_data)
        GraphLab.run_logistic_regression_split(train_data, test_data)
        GraphLab.run_random_forest_split(train_data, test_data)

        # K-fold
        folds = GraphLab.kfold_data(encrypted_stock_market_data)
        GraphLab.run_logistic_regression_kfold(folds)
        GraphLab.run_random_forest_kfold(folds)

    @staticmethod
    def split_data(dataset):
        train_data, test_data = dataset.random_split(1 - SPLIT_RATIO, seed=SEED)
        return train_data, test_data

    @staticmethod
    def kfold_data(dataset):
        dataset = graphlab.cross_validation.shuffle(dataset, random_seed=SEED)
        folds = graphlab.toolkits.cross_validation.KFold(dataset, K_FOLD)
        return folds

    @staticmethod
    def get_features(data):
        features = data.column_names()
        target_feature = features[len(features) - 1]
        features = features[:-1]
        return target_feature, features

    @staticmethod
    def run_logistic_regression_split(train_data, test_data):
        print("\nRunning Logistic Regression with Split Validation...")
        target_feature, features = GraphLab.get_features(train_data)
        logistic_regression = graphlab.logistic_classifier.create(train_data, target=target_feature, features=features)
        GraphLab.split_validation(test_data, target_feature, logistic_regression)

    @staticmethod
    def run_random_forest_split(train_data, test_data):
        print("\Running Random Forest with Split Validation...")
        target_feature, features = GraphLab.get_features(train_data)
        random_forest = graphlab.random_forest_classifier.create(train_data, target=target_feature, features=features,
                                                                 random_seed=SEED)
        GraphLab.split_validation(test_data, target_feature, random_forest)

    @staticmethod
    def run_logistic_regression_kfold(folds):
        print("\nRunning Logistic Regression with Cross Validation...")
        GraphLab.cross_validation(folds, GraphLab.Algorithm_IDENTIFIER.LOGISTIC_REGRESSION)

    @staticmethod
    def run_random_forest_kfold(folds):
        print("\nRunning Random Forest with Cross Validation...")
        GraphLab.cross_validation(folds, GraphLab.Algorithm_IDENTIFIER.RANDOM_FOREST)

    @staticmethod
    def split_validation(test_data, target_feature, algorithm):
        prediction = algorithm.predict(test_data)
        accuracy = graphlab.evaluation.accuracy(test_data[target_feature], prediction)
        precision = graphlab.evaluation.precision(test_data[target_feature], prediction)
        print("Accuracy Score: ", accuracy)
        print("Precision Score: ", precision)

    @staticmethod
    def cross_validation(folds, algorithm_identifier):
        print("\nRunning Cross Validation...")
        accuracy_sum = 0
        precision_sum = 0

        for fold in range(0, K_FOLD):
            (train_data, test_data) = folds[fold]
            target_feature, features = GraphLab.get_features(train_data)

            if algorithm_identifier is GraphLab.Algorithm_IDENTIFIER.LOGISTIC_REGRESSION:
                algorithm = graphlab.logistic_classifier.create(train_data, target=target_feature,
                                                                features=features)
            else:
                algorithm = graphlab.random_forest_classifier.create(train_data, target=target_feature,
                                                                     features=features, random_seed=SEED)

            prediction = algorithm.predict(test_data)
            accuracy = graphlab.evaluation.accuracy(test_data[target_feature], prediction)
            accuracy_sum += accuracy
            precision = graphlab.evaluation.precision(test_data[target_feature], prediction)
            precision_sum += precision

        print("Average Accuracy Score: ", accuracy_sum / K_FOLD)
        print("Average Precision Score: ", precision_sum / K_FOLD)


class H2O(Framework):
    @classmethod
    def init(cls):
        h2o.init()
        h2o.remove_all()

    @staticmethod
    def run_skin_dataset():
        print("\n\nRunning Skin Dataset...")
        file_path = load_file(DATASET_2)
        skin_dataset = h2o.import_file(file_path)

        # Set Binary category to 0 and 1 instead of 1 and 2
        skin_dataset[skin_dataset.dim[COLUMNS_INDEX] - 1] = skin_dataset[skin_dataset.dim[COLUMNS_INDEX] - 1] - 1

        # Set column as categorical
        skin_dataset[skin_dataset.dim[COLUMNS_INDEX] - 1] = skin_dataset[skin_dataset.dim[COLUMNS_INDEX] - 1].asfactor()

        col_X = skin_dataset.col_names[:-1]
        col_y = skin_dataset.col_names[-1]

        H2O.run_algorithms(col_X, col_y, skin_dataset)

    @staticmethod
    def run_algorithms(X, y, dataset):
        H2O.run_logistic_regression(X, y, dataset)
        H2O.run_naive_bayes(X, y, dataset)
        H2O.run_random_forest(X, y, dataset)

    @staticmethod
    def run_encrypted_stock_market_dataset():
        print("\n\nRunning Encrypted Stock Market Dataset...")
        file_path = load_file(DATASET_1)
        encrypted_stock_market_dataset = h2o.import_file(file_path)

        # Set Target class column as categorical
        target_class_column = encrypted_stock_market_dataset.dim[COLUMNS_INDEX] - 1
        encrypted_stock_market_dataset[target_class_column] = encrypted_stock_market_dataset[
            target_class_column].asfactor()

        col_X = encrypted_stock_market_dataset.col_names[:-1]
        col_y = encrypted_stock_market_dataset.col_names[-1]

        H2O.run_algorithms(col_X, col_y, encrypted_stock_market_dataset)

    @staticmethod
    def run_logistic_regression(X, y, dataset):
        print("\nRunning Logistic Regression...")
        logistic_regression_split = H2OGeneralizedLinearEstimator(family="binomial")
        H2O.split_validation(logistic_regression_split, X, y, dataset)

        logistic_regression_cross_validation = H2OGeneralizedLinearEstimator(family="binomial", nfolds=K_FOLD,
                                                                             seed=SEED)
        H2O.cross_validation(logistic_regression_cross_validation, X, y, dataset)

    @staticmethod
    def run_naive_bayes(X, y, dataset):
        print("\nRunning Naive Bayes...")
        naiveBayes_split = H2ONaiveBayesEstimator()
        H2O.split_validation(naiveBayes_split, X, y, dataset)
        naiveBayes_cross_validation = H2ONaiveBayesEstimator(nfolds=K_FOLD, seed=SEED)
        H2O.cross_validation(naiveBayes_cross_validation, X, y, dataset)

    @staticmethod
    def run_random_forest(X, y, dataset):
        print("\nRunning Random Forest...")
        random_forest_split = H2ORandomForestEstimator(seed=SEED)
        H2O.split_validation(random_forest_split, X, y, dataset)
        random_forest_cross_validation = H2ORandomForestEstimator(nfolds=K_FOLD, seed=SEED)
        H2O.cross_validation(random_forest_cross_validation, X, y, dataset)

    @staticmethod
    def split_validation(algorithm, X, y, dataset):
        print("Running Split 70/30...")
        test, train = dataset.split_frame([SPLIT_RATIO], seed=SEED)
        algorithm.train(X, y, training_frame=train, validation_frame=test)
        H2O.evaluate(algorithm)

    @staticmethod
    def cross_validation(algorithm, X, y, dataset):
        print("Running Cross Validation...")
        algorithm.train(X, y, training_frame=dataset)
        H2O.evaluate(algorithm)

    @staticmethod
    def evaluate(algorithm):
        print("Accuracy Score:", algorithm.accuracy())
        print("Precision Score:", algorithm.precision())


class ScikitLearn(Framework):
    @classmethod
    def init(cls):
        pass

    @staticmethod
    def run_encrypted_stock_market_dataset():
        print("\n\nRunning Encrypted Stock Market Dataset...")
        encrypted_stock_market_dataset = load_dataset_as_np_array(DATASET_1, ",")

        features_count = encrypted_stock_market_dataset.shape[COLUMNS_INDEX] - 1
        X = encrypted_stock_market_dataset[:, :features_count]
        y = encrypted_stock_market_dataset[:, features_count]

        ScikitLearn.run_algorithms(X, y)

    @staticmethod
    def run_skin_dataset():
        print("\n\nRunning Skin dataset...")
        skin_dataset = load_dataset_as_np_array(DATASET_2, "\t", header=None)

        features_count = skin_dataset.shape[COLUMNS_INDEX] - 1
        X = skin_dataset[:, :features_count]
        y = skin_dataset[:, features_count]

        ScikitLearn.run_algorithms(X, y)

    @staticmethod
    def run_algorithms(X, y):
        ScikitLearn.run_logistic_regression(X, y)
        ScikitLearn.run_naive_bayes(X, y)
        ScikitLearn.run_random_forest(X, y)

    @staticmethod
    def run_logistic_regression(X, y):
        print("\nRunning Logistic Regression...")
        logistic_regression = linear_model.LogisticRegression()
        ScikitLearn.evaluate_algorithm(logistic_regression, X, y)

    @staticmethod
    def run_naive_bayes(X, y):
        print("\nRunning Naive Bayes...")
        gaussian_naive_bayes = naive_bayes.GaussianNB()
        ScikitLearn.evaluate_algorithm(gaussian_naive_bayes, X, y)

    @staticmethod
    def run_random_forest(X, y):
        print("\nRunning Random Forest...")
        random_forest = ensemble.RandomForestClassifier(random_state=SEED)
        ScikitLearn.evaluate_algorithm(random_forest, X, y)

    @staticmethod
    def evaluate_algorithm(algorithm, X, y):
        ScikitLearn.split_validation(algorithm, X, y)
        ScikitLearn.cross_validation(algorithm, X, y)

    @staticmethod
    def split_validation(algorithm, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT_RATIO)
        algorithm.fit(X_train, y_train)
        print("70/30 Split Score: ", algorithm.score(X_test, y_test))

        predictions = algorithm.predict(X_test)
        ScikitLearn.evaluate(y_test, predictions)

    @staticmethod
    def cross_validation(algorithm, X, y):
        k_fold = KFold(n_splits=K_FOLD, random_state=SEED)
        scores = cross_val_score(algorithm, X, y, cv=k_fold)
        print("10-Fold Cross Validation: ", scores)
        print("Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

        predictions = cross_val_predict(algorithm, X, y, cv=k_fold)
        ScikitLearn.evaluate(y, predictions)

    @staticmethod
    def evaluate(y, predictions):
        accuracy = metrics.accuracy_score(y, predictions)
        print("Accuracy Score:", accuracy)

        precision = metrics.precision_score(y, predictions)
        print("Precision Score:", precision)


def load_file(file_key):
    file_manager = DatasetFileManager()
    return file_manager.get_dataset_path(file_key)


def load_dataset_as_graphlab_data_frame(file_key, delimiter):
    file_path = load_file(file_key)
    dataset = graphlab.SFrame.read_csv(file_path, header=None, sep=delimiter)
    return dataset


def load_dataset_as_data_frame(file_key, delimiter, header="infer"):
    file_path = load_file(file_key)
    dataset = pd.read_csv(file_path, header=header, sep=delimiter)
    return dataset


def load_dataset_as_np_array(file_key, delimiter, header="infer"):
    data_frame = load_dataset_as_data_frame(file_key, delimiter, header=header)
    return data_frame.values


def main():
    ScikitLearn.run()
    H2O.run()
    GraphLab.run()


if __name__ == "__main__":
    main()
