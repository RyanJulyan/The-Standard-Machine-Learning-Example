# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LinearRegression

from brokers.forecast_model.i_forecast_model import IForecastModel


@dataclass
class SimpleLinearRegression(IForecastModel):
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

    def fit(self, data: Iterable[float]) -> SimpleLinearRegression:
        self.data = np.array(data)
        X = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data
        self.model.fit(X, y)
        self.backcast_values = self.model.predict(X)

        return self

    def predict(self, steps: int) -> SimpleLinearRegression:
        X_future = np.arange(len(self.data), len(self.data) + steps).reshape(-1, 1)
        self.forecast_values = self.model.predict(X_future)

        if self.disallow_negative:
            self.backcast_values = np.maximum(0, self.backcast_values)
            self.forecast_values = np.maximum(0, self.forecast_values)

        return self


# %%
# Example usage
if __name__ == "__main__":
    data = [1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 2]
    steps = 13

    model = SimpleLinearRegression()
    model.fit(data)
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
