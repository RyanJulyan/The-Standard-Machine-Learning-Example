# %%
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class IForecastModel(ABC):
    """
    Represents an abstract forecasting model that provides methods for training and predicting on time series data.

    Args:
        data (np.ndarray): The forecast data object to be used for training and prediction.
        model (Any): The current instantiated forecast model class
        forecast_values (Iterable[float]): The forecasted values.
        backcast_values (Iterable[float]): The backcasted values.
    """

    data: np.ndarray
    model: Any = None
    forecast_values: np.ndarray = np.array([])
    backcast_values: np.ndarray = np.array([])

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> IForecastModel:
        """
        Fits the forecasting model to the training data.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            IForecastingModel: A reference to the current instance of the forecasting model.
        """

        raise NotImplementedError("`fit` method not implemented")

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> IForecastModel:
        """
        Predicts on the testing set.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            IForecastingModel: A reference to the current instance of the forecasting model.
        """

        raise NotImplementedError("`predict` method not implemented")


# %%
