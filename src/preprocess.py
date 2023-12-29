import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract base class defining a common interface for different data handling strategies.
    This class represents the Strategy interface in the Strategy Design Pattern, allowing
    for the implementation of various data handling algorithms that can be interchanged
    within the context class.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method to be implemented by concrete strategy classes.
        Defines how data should be processed by different strategies.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed.

        Returns:
        Union[pd.DataFrame, pd.Series]: The processed DataFrame or Series.
        """
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Concrete implementation of the DataStrategy for preprocessing data.
    This class encapsulates the algorithm for data preprocessing including operations like
    dropping specific columns, filling missing values, and filtering data types.
    It's a concrete strategy in the Strategy Design Pattern.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implements data preprocessing strategy.

        Parameters:
        data (pd.DataFrame): The DataFrame to be preprocessed.

        Returns:
        pd.DataFrame: The preprocessed DataFrame.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Concrete implementation of the DataStrategy for dividing data into train and test sets.
    This class encapsulates the algorithm for splitting the data into training and testing sets.
    It's another concrete strategy in the Strategy Design Pattern.
    """

    def handle_data(
            self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Implements data division strategy.

        Parameters:
        data (pd.DataFrame): The DataFrame to be split.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing sets.
        """
        try:
            X = data.drop("review_score", axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Context class in the Strategy Design Pattern which is configured with a DataStrategy object.
    This class is responsible for delegating the data handling to the current strategy object
    and can switch strategies dynamically at runtime.

    Attributes:
    df (pd.DataFrame): The DataFrame to be processed.
    strategy (DataStrategy): The current strategy for processing the data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """
        Initializes the DataCleaning class with a specific data handling strategy.

        Parameters:
        data (pd.DataFrame): The DataFrame to be processed.
        strategy (DataStrategy): The strategy to use for processing the data.
        """
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Processes the data using the current strategy.

        Returns:
        Union[pd.DataFrame, pd.Series]: The processed data.
        """
        return self.strategy.handle_data(self.df)
