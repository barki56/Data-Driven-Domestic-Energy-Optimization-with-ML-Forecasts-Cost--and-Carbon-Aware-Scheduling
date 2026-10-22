# Domestic Energy Forecasting & Optimisation (MSc Dissertation)

My MSc Data Science dissertation at the University of Greenwich (Distinction, project mark 87%). I built a system that forecasts household electricity use and then recommends the cheapest and lowest-carbon time to run appliances like washing machines, dishwashers or EV chargers.

The idea came from a simple problem. Time-of-use tariffs make electricity cheaper at certain times of day, and the grid is cleaner at certain times too, but most people have no practical way to act on that. I wanted to turn a forecast into a clear, specific recommendation, like "run your dishwasher at 9pm".

## What it does

1. **Forecasts** each household's daily electricity use from its past usage, the weather and the property itself.
2. **Explains** the forecast with SHAP, so it's clear what is driving the prediction.
3. **Recommends** the best hour to run an appliance within a window the user chooses, balancing cost and carbon based on their own priorities.
4. **Serves** all of this through a FastAPI web service and an interactive Streamlit dashboard.

## The data

I combined three public UK sources:

- **Smart meter readings** for London households, 2011–2014 ([London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)), filtered to Camden and Tower Hamlets
- **Energy Performance Certificates (EPC)**: floor area, energy rating, property type, built form and construction age ([GOV.UK](https://www.gov.uk/find-energy-certificate))
- **Weather**: temperature, rain, humidity, cloud cover and wind (Met Office)

Each household was linked to its EPC record and to the weather for each day, giving roughly 1.2GB of data. The data isn't included in this repo because of its size, but all sources are public.

## How I built it

**Feature engineering.** On top of the raw weather and property data, I created:
- lagged usage (1, 2, 3 and 7 days before) and 7-day rolling mean and standard deviation, calculated separately for each household
- seasonal features using sine and cosine of the day of the year, plus day of week, month and season
- interaction features such as temperature × humidity and floor area × energy rating

**Avoiding data leakage.** This was the part I was most careful with. The data was split by time, not randomly: the model trained on **2011–2013** and was tested only on **2014**, which it had never seen. Cross-validation used `TimeSeriesSplit`, and the preprocessing was fitted on training data only. A random split would have let the model "see the future" and given misleadingly good results.

**Modelling.** I tuned XGBoost and LightGBM separately with `RandomizedSearchCV`, then combined them in a stacked ensemble with an ElasticNet meta-model that learns how much to trust each one.

**Explainability.** I used SHAP to check that the model was learning sensible patterns rather than noise.

## Results

On the 2014 hold-out data:

| Metric | Result |
|---|---|
| R² | **0.88** |
| MAE | **1.87 kWh per day** |

The stacked ensemble did slightly better than either model on its own.

**What drives the predictions (SHAP):**
- **Recent usage** (yesterday's usage and the 7-day average) was the strongest predictor
- **Temperature and season**: cold days push usage up, mainly from heating
- **Time patterns**: day of week and time of use reflect people's routines
- **The property**: bigger homes and worse energy ratings mean higher usage

**Per-household check.** I also wrote a validation script (`validate_lcid_2014.py`) to test the model one household at a time. Accuracy varied a lot between homes. Some were predicted very closely (MAE around 0.13 kWh), while others with more irregular routines were much harder (MAE around 2.5 kWh). That's an important limitation, because a single model doesn't fit every household equally well.

## The optimisation

For a chosen appliance, date and time window, the system:
1. predicts the household's baseline usage for each hour in the window
2. adds the appliance's energy use
3. calculates cost and carbon using a time-of-use price and grid carbon intensity
4. scores each hour with a weighted combination of cost and carbon (the user sets the balance)
5. recommends the lowest-scoring hour

In simulations under a time-of-use tariff, shifting flexible appliance use this way pointed to potential savings of around 15–20% on cost and CO₂ for a typical household.

## Dashboard

The Streamlit dashboard has four sections:
- **Overview & Trends**: consumption, cost and carbon over time, plus a correlation heatmap
- **Individual Prediction**: enter weather and property details and get a live forecast from the API
- **Optimisation Recommendations**: pick an appliance and time window to get the best hour, with an hourly cost and carbon breakdown
- **Residuals Analysis**: where the model's errors are, and whether they're biased

<!-- Add screenshots here, e.g. ![Dashboard](screenshots/dashboard.png) -->

## Running it

You'll need the trained model files in `api/models/` (`best_xgb_regressor.pkl`, `best_lgbm_regressor.pkl`, `meta_model.pkl`, `preprocessor.pkl`, `model_features.pkl`).

```bash
pip install -r requirements.txt

# Start the API (docs at http://127.0.0.1:8000/docs)
cd api
uvicorn energy_api:app --reload --port 8000

# In a second terminal, start the dashboard
cd dashboard
streamlit run energy_dashboard.py
```

API endpoints:
- `POST /predict`: predicted kWh for a given set of inputs
- `POST /optimize_appliance`: best hour to run an appliance within a time window
- `GET /health`: checks the API is running and the models are loaded

## Limitations and what I'd do next

- **Prices and carbon intensity in the API are simulated.** The next step would be connecting real data, such as the National Grid ESO Carbon Intensity API and a dynamic tariff like Octopus Agile.
- **Lag features are entered manually in the demo.** In a real system they'd come straight from the household's smart meter history.
- **Accuracy varies between households**, so a per-household or clustered model could help with harder-to-predict homes.
- **It only covers London households from 2011–2014**, before widespread solar panels, heat pumps and EVs, so it would need retraining on newer data.
- **The savings are simulated.** A real trial with households would show what people actually save and whether they follow the recommendations.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, FastAPI, Streamlit, Plotly, Matplotlib, Seaborn
