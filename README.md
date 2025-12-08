Domestic Energy Optimization with Machine Learning

Cost- and carbon-aware forecasting & appliance scheduling

This repository contains the full implementation of my MSc Data Science project: an end-to-end machine-learning system that predicts household electricity consumption and recommends optimal times to run appliances based on tariff rates and grid carbon intensity.
This system integrates smart-meter electricity usage, weather data, and Energy Performance Certificate (EPC) building features to forecast short-term energy consumption and optimise appliance scheduling.

It includes:
	•	A complete ML pipeline (cleaning → feature engineering → modelling → evaluation)
	•	High-accuracy forecasting using XGBoost, LightGBM, and a stacked ElasticNet ensemble
	•	Full SHAP explainability for feature importance & model trust
	•	A FastAPI backend for real-time predictions and appliance optimisation
	•	A Streamlit dashboard for visualisation, forecasting, and scheduling recommendations

  Key Features

 Machine Learning Pipeline
	•	Time-series split (2011–2013 train → 2014 test)
	•	Rich feature engineering: temporal, weather, EPC traits, cyclic encodings, lag features, rolling statistics
	•	Stacked ensemble model with R² ≈ 0.88–0.90 on unseen future data
	•	SHAP plots for model interpretability

 FastAPI Backend

Endpoints include:
	•	/health – service status
	•	/predict – returns predicted kWh consumption
	•	/optimize_appliance – recommends lowest cost/carbon hour within user’s time window

 Streamlit Dashboard
	•	Historic energy trends and heatmaps
	•	Interactive “what-if” prediction page
	•	Appliance scheduling with cost–carbon weighting
	•	Residuals & error analysis for transparency

.
├── data/
│   └── optimisation_dataset_with_predictions.csv
├── models/
│   ├── preprocessor.pkl
│   ├── best_xgb_regressor.pkl
│   ├── best_lgbm_regressor.pkl
│   └── meta_model.pkl
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── notebooks/
│   └── model_training.ipynb
└── README.md

Installation
  git clone <your-repo-link>
  cd <repo-name>
  pip install -r requirements.txt

Run the API
  git clone <your-repo-link>
  cd <repo-name>
  pip install -r requirements.txt

Run the Dashboard
  streamlit run dashboard/app.py

SHAP analysis highlights:
	•	Lagged usage
	•	Temperature & seasonal features
	•	Hour-of-day patterns
	•	Building size & EPC rating

  These align with human-interpretable drivers of household energy consumption.

Impact
  Simulations show potential savings of 15–20% in cost and CO₂ emissions by shifting appliance usage to cleaner, cheaper periods.

Future Work
	•	Multi-appliance optimisation
	•	Real carbon-intensity API integration
	•	Real-time data streaming
	•	Deployment on cloud (AWS/GCP/Azure)

  
