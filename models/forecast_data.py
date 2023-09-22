from dataclasses import dataclass
from datetime import datetime


@dataclass
class ForecastData:
    """
    Represents a forecast data object with a time series index and corresponding numerical values.

    Args:
        date (datetime): A time series index representing the dates of the values.
        value (float): A numerical data array representing the values corresponding to each date.
    """

    date: datetime
    value: float
