import logging
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models.

    This class serves as a base for different machine learning models.
    It defines a common interface for training models, ensuring that all subclasses
    implement the train method.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Abstract method to train the model.

        Subclasses should implement this method to train the model using
        the provided training data.

        Args:
            X_train: Training feature dataset.
            y_train: Training target dataset.
        """
        pass


class LinearRegressionModel(Model):
    """
    Concrete implementation of the Model class for Linear Regression.

    This class implements the train method specifically for training a linear
    regression model using the scikit-learn library.
    """

    def train(self, X_train, y_train, **kwargs):
        """
        Trains a Linear Regression model with the given training data.

        Uses the LinearRegression class from scikit-learn and allows additional
        keyword arguments to be passed to the LinearRegression constructor.

        Args:
            X_train: Training feature dataset.
            y_train: Training target dataset.
            **kwargs: Additional keyword arguments to pass to the LinearRegression constructor.

        Returns:
            A trained LinearRegression model.

        Raises:
            Exception: If an error occurs during model training.
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e
