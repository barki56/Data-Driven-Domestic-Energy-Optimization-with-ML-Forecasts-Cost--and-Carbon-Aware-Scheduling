## Domestic Energy Forecasting & Optimization (MSc Dissertation)

An end-to-end machine learning system that forecasts household electricity usage and
optimizes appliance scheduling to reduce cost and carbon emissions.

MSc Data Science (Distinction) – University of Greenwich  
Dataset size: 1.2GB smart-meter + weather + EPC data (CSV) 
Models: XGBoost, LightGBM, CatBoost  

### Why this project matters
Residential energy consumption is a major contributor to cost volatility and carbon emissions.
This project demonstrates how machine learning can be used to deliver interpretable,
data-driven recommendations for households and energy stakeholders.

### My contribution
- Designed a leakage-free time-series ML pipeline with strict temporal validation
- Engineered temporal, weather, EPC and lag-based features
- Trained and tuned ensemble models achieving R² ≈ 0.88–0.90 on unseen data
- Applied SHAP for transparent model explainability
- Built a FastAPI prediction & optimization API
- Developed a Streamlit dashboard for scenario analysis
