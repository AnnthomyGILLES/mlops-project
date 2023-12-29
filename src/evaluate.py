import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance.
    This class serves as a base for different model evaluation strategies.
    Subclasses should implement the calculate_score method.
    """

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Abstract method to calculate and return an evaluation metric based on
        true values and predicted values.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The calculated evaluation metric.
        """
        pass


class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE) for model evaluation.
    Inherits from the Evaluation class and implements the calculate_score method
    to compute the MSE between true and predicted values.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate and return the Mean Squared Error (MSE) between y_true and y_pred.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The Mean Squared Error (MSE) value.
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE value: {mse}")
            return mse
        except Exception as e:
            logging.error("Error in MSE calculation: " + str(e))
            raise e


class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 Score (Coefficient of Determination) for model evaluation.
    Inherits from the Evaluation class and implements the calculate_score method
    to compute the R2 Score between true and predicted values.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate and return the R2 Score (Coefficient of Determination) between y_true and y_pred.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The R2 Score value.
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score value: {r2}")
            return r2
        except Exception as e:
            logging.error("Error in R2 Score calculation: " + str(e))
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE) for model evaluation.
    Inherits from the Evaluation class and implements the calculate_score method
    to compute the RMSE between true and predicted values.
    """

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate and return the Root Mean Squared Error (RMSE) between y_true and y_pred.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: The Root Mean Squared Error (RMSE) value.
        """
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE value: {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error in RMSE calculation: " + str(e))
            raise e
