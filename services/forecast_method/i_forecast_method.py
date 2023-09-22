# %%
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from brokers.forecast_model.i_forecast_model import (
    IForecastModel as IForecastModelBroker,
)
from services.forecast_model.i_forecast_model import (
    IForecastModel as IForecastModelService,
)


@dataclass
class IForecastMethod(ABC):
    """
    Represents an abstract forecasting model that provides methods for training and predicting on time series data.

    Args:
        data (np.ndarray): The forecast data object to be used for training and prediction.
        model (Any): The current instantiated forecast model class
        forecast_values (Iterable[float]): The forecasted values.
        backcast_values (Iterable[float]): The backcasted values.
    """

    data: np.ndarray
    model_name: str
    model: Optional[Union[IForecastModelBroker, IForecastModelService]] = None
    forecast_values: np.ndarray = np.array([])
    backcast_values: np.ndarray = np.array([])

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> IForecastMethod:
        """
        Fits the forecasting model to the training data.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            IForecastingModel: A reference to the current instance of the forecasting model.
        """

        raise NotImplementedError("`fit` method not implemented")

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> IForecastMethod:
        """
        Predicts on the testing set.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.

        Returns:
            IForecastingModel: A reference to the current instance of the forecasting model.
        """

        raise NotImplementedError("`predict` method not implemented")


# %%
