# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

from services.forecast_model.i_forecast_model import IForecastModel


@dataclass
class SimpleLinearRegressionWithTrendAndSeasonality(IForecastModel):
    seasonal_period: Optional[int] = 1
    data: np.ndarray = np.array([])
    model: Any = None
    forecast_values: np.ndarray = np.array([])
    backcast_values: np.ndarray = np.array([])
    disallow_negative: bool = True

    def __init__(
        self,
        disallow_negative: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        self.disallow_negative = disallow_negative
        self.model = LinearRegression(*args, **kwargs)

    def fit(
        self, data: np.ndarray, seasonal_period: Optional[int] = None
    ) -> SimpleLinearRegressionWithTrendAndSeasonality:
        self.seasonal_period = (
            self.seasonal_period if seasonal_period is None else seasonal_period
        )
        self.data = np.array(data)

        n = len(self.data)
        x = np.arange(n)
        X = np.column_stack(
            [
                x,
                np.sin(2 * np.pi * x / self.seasonal_period),
                np.cos(2 * np.pi * x / self.seasonal_period),
            ]
        )

        self.model.fit(X, self.data)

        # Generate backcasts
        self.backcast_values = self.model.predict(X)

        return self

    def predict(self, steps: int) -> SimpleLinearRegressionWithTrendAndSeasonality:
        """Generate forecasts for a given number of steps.

        Args:
            steps (int): The number of steps to forecast.
        """
        if self.data is None:
            raise ValueError("Model not fitted yet.")

        n = len(self.data)

        # Generate forecasts
        x_future = np.arange(n, n + steps)
        X_future = np.column_stack(
            [
                x_future,
                np.sin(2 * np.pi * x_future / self.seasonal_period),
                np.cos(2 * np.pi * x_future / self.seasonal_period),
            ]
        )
        self.forecast_values = self.model.predict(X_future)

        if self.disallow_negative:
            self.backcast_values = np.maximum(0, self.backcast_values)
            self.forecast_values = np.maximum(0, self.forecast_values)

        return self


# %%
# Example usage
if __name__ == "__main__":
    data = [1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 2]
    seasonal_period = 12  # None  #
    steps = 13

    model = SimpleLinearRegressionWithTrendAndSeasonality()
    model.fit(data, seasonal_period=seasonal_period)
    model.predict(steps=steps)

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
