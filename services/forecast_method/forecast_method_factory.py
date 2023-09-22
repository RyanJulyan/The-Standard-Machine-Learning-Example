# %%
from __future__ import annotations
from dataclasses import dataclass
import pprint
from typing_extensions import Literal
from typing import Any, Dict, Optional, Union

import numpy as np

from services.forecast_method.i_forecast_method import IForecastMethod
from brokers.forecast_model.i_forecast_model import (
    IForecastModel as IForecastModelBroker,
)
from services.forecast_model.i_forecast_model import (
    IForecastModel as IForecastModelService,
)
from brokers.forecast_model.arima_model import ARIMAModel
from brokers.forecast_model.simple_linear_regression import SimpleLinearRegression
from services.forecast_model.simple_linear_regression_with_trend_and_seasonality import (
    SimpleLinearRegressionWithTrendAndSeasonality,
)


@dataclass
class ForecastMethodFatory(IForecastMethod):
    """
    Represents an abstract forecasting model that provides methods for training and predicting on time series data.

    Args:
        data (np.ndarray): The forecast data object to be used for training and prediction.
        model_name (str): Name of the forecast model to use and load into `model`
        model (Any): The current instantiated forecast model class
        forecast_values (Iterable[float]): The forecasted values.
        backcast_values (Iterable[float]): The backcasted values.
    """

    data: np.ndarray
    model_name: Union[
        Literal[
            "ARIMAModel",
            "SimpleLinearRegression",
            "SimpleLinearRegressionWithTrendAndSeasonality",
        ],
        str,
    ]
    model: Optional[Union[IForecastModelBroker, IForecastModelService]] = None
    forecast_values: np.ndarray = np.array([])
    backcast_values: np.ndarray = np.array([])

    def __init__(
        self,
        model_name: Union[
            Literal[
                "ARIMAModel",
                "SimpleLinearRegression",
                "SimpleLinearRegressionWithTrendAndSeasonality",
            ],
            str,
        ],
        custom_model_factory: Dict[
            str, Union[IForecastModelBroker, IForecastModelService]
        ] = {},
        *args: Any,
        **kwargs: Any,
    ):
        default_model_factory: Dict[
            str, Union[IForecastModelBroker, IForecastModelService]
        ] = {
            "ARIMAModel": ARIMAModel,
            "SimpleLinearRegression": SimpleLinearRegression,
            "SimpleLinearRegressionWithTrendAndSeasonality": SimpleLinearRegressionWithTrendAndSeasonality,
        }
        self.model_factory: Dict[
            str, Union[IForecastModelBroker, IForecastModelService]
        ] = {
            **default_model_factory,
            **custom_model_factory,
        }

        self.model = self.model_factory[model_name](*args, **kwargs)

    def fit(self, *args: Any, **kwargs: Any) -> ForecastMethodFatory:
        """
        Fits the forecasting model to the training data.

        Returns:
            IForecastingModel: A reference to the current instance of the forecasting model.
        """

        self.model.fit(*args, **kwargs)

        self.backcast_values = self.model.backcast_values

        return self

    def predict(self, *args: Any, **kwargs: Any) -> ForecastMethodFatory:
        """
        Predicts on the testing set.

        Returns:
            IForecastingModel: A reference to the current instance of the forecasting model.
        """

        self.model.predict(*args, **kwargs)

        self.backcast_values = self.model.backcast_values
        self.forecast_values = self.model.forecast_values

        return self


# %%
# Example usage
if __name__ == "__main__":
    data = [1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 2]
    steps = 13

    custom_model_factory = {
        "MyCustomModel": SimpleLinearRegressionWithTrendAndSeasonality,
    }

    model = ForecastMethodFatory(
        model_name="SimpleLinearRegressionWithTrendAndSeasonality",
        custom_model_factory=custom_model_factory,
    )
    model.fit(data, seasonal_period=6)
    model.predict(steps=steps)

    print("Models:")
    pprint.pprint(model.model_factory)
    print()
    print(f"Data: {data}")
    print(f"Backcasts: {model.backcast_values}")
    print(f"Forecasts for next {steps} steps: {model.forecast_values}")

    # %%
    # Plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(data)), data, color="green", label="Data")
    plt.bar(
        range(len(data), len(data) + steps),
        model.forecast_values,
        color="yellow",
        label="Forecasts",
    )
    plt.plot(model.backcast_values, color="red", label="Model")
    plt.axvline(x=len(data) - 0.5, color="black", linestyle="--")
    plt.xlabel("Period")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# %%
