# The-Standard-Machine-Learning-Example
Trying to apply [The-Standard](https://github.com/hassanhabib/The-Standard/) to a Machine Learning Example

## Original question:
>Hi @Hassan Habib and community. 
> 
> I am writing a forecasting engine, and want to clarify where to place specific classes to be more standard complient.
> 
> TLDR: I have a mix of external and internal statistical models and want to call them from the same exposure point for a decent flow. I Assume: Exposure Point > Orchestration service > factory service > foundation service (either bespoke statistical model  internal service or broker wrapper service)  > broker(external model library) is my assumption. Is that the correct Flow or did I miss something?
> 
> My forecasting engine includes Brokers with an IForecastModel. The fit and predict methods should be implemented by other classes, such as ARIMAModel(IForecastModel). I believe the broker role is necessary because I use external statistics libraries with clear expectations, such as statsmodels.tsa.arima.model, which I want to interface with and ensure consistency in my engine implementation.
> 
> I want to implement a custom Statistical Model, SuperAwesomeCustomForecastModel123. As an internal statistical model, I believe it should be a Service as I will implement it from scratch. I want my forecasting engine to be consistent so, is that placement correct? Should they be categorized as a broker instead of a service for consistency?
> 
> Assuming proper placement, should a Service be created for each other Broker that acts as a wrapper for an external library? Creating a broker for my internal service (statistical model) seems like a poor approach and could cause misunderstanding.
> 
> If  SuperAwesomeCustomForecastModel123 is a Broker, importing  from the same level as the other Brokers seems incorrect, as I assume the flow forward applies to brokers as described in the services section. [https://github.com/hassanhabib/The-Standard/blob/master/2.%20Services/2.%20Services.md#2025-flow-forward](https://github.com/hassanhabib/The-Standard/blob/master/2.%20Services/2.%20Services.md#2025-flow-forward)◊
> 
> Sorry for the long post. I am keen to better understand and hope I made sense.

