# %%
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from brokers.forecast_model.i_forecast_model import IForecastModel


@dataclass
class ARIMAModel(IForecastModel):
    data: np.ndarray = np.array([])
    model: Any = None
    forecast_values: np.ndarray = np.array([])
    backcast_values: np.ndarray = np.array([])
    disallow_negative: bool = True

    def __init__(
        self,
        order: tuple = (1, 1, 1),  # Default ARIMA order (p, d, q)
        disallow_negative: bool = True,
    ):
        self.order = order
        self.model = ARIMA
        self.disallow_negative = disallow_negative

    def fit(
        self,
        data: Iterable[float],
        order: Optional[tuple] = None,  # Default ARIMA order (p, d, q)
        *args: Any,
        **kwargs: Any,
    ) -> ARIMAModel:
        self.order = self.order if order is None else order
        self.data = np.array(data)
        self.model = self.model(self.data, order=self.order, *args, **kwargs)
        self.model_fit = self.model.fit()
        self.backcast_values = self.model_fit.fittedvalues

        return self

    def predict(self, steps: int) -> ARIMAModel:
        self.forecast_values = self.model_fit.forecast(steps=steps)

        if self.disallow_negative:
            self.backcast_values = np.maximum(0, self.backcast_values)
            self.forecast_values = np.maximum(0, self.forecast_values)

        return self


# %%
# Example usage
if __name__ == "__main__":
    data = [1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 1, 2, 10, 4, 4, 4, 3, 5, 2, 1, 2, 1, 2]
    steps = 13

    model = ARIMAModel(order=(1, 1, 1))
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
