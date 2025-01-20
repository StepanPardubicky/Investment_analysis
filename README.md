# Portfolio Analysis and Management App

The repository contains a Streamlit application developed as a semestral project for the course *BI-PYT Programování v Pythonu*.
Main parts of the portfolio app are:
1. **Exploratory Data Analysis**: Perform exploratory data analysis (EDA) on the stock and bond returns. This includes visualizing stock value development, calculating descriptive statistics, and identifying missing and zero values.
2. **Portfolio Analysis**: Perform mean-variance portfolio analysis on the stock and bond returns. This includes calculating the efficient frontier, capital market line, and tangent portfolio, as well as visualizing sample portfolio weights.
3. **Portfolio Value Evolution**: Visualize the evolution of the portfolio constructed via chosen strategy over time. This includes calculating daily portfolio values, portfolio weights, and comparing different investment strategies.
4. **Anomaly Detection**: Perform anomaly detection on stock and bond returns using autoencoder models. Autoencoder models for the task are manually trained

The application encompasses majority of the developed functionality, but the developed functions have further applications and allow for additional functionality/app development.

## Dependencies
On top of the basic dependencies for the course *BI-PYT* (can be found on the course website - pip install jupyter numpy pandas matplotlib pillow pytest), additional dependencies are required. They can be found in `requirements.txt` file:
 - yfinance==0.2.51
 - scipy==1.14.1
 - statsmodels==0.14.4
 - seaborn==0.13.2
 - streamlit==1.41.1
 - pyod==2.0.3
 - gurobi_optimods==2.3.1
 - torch==2.5.1
 - tensorflow==2.18.0
 - keras==3.7.0
Suggested approach is to follow guidelines on website for creating virtual enviroment, activating it and the then runnung the command: `pip install -r requirements.txt`

## How to start the App (Streamlit)
To start the app, navigate to the directory semestral and run the fillowing command `streamlit run portfolio_analysis_app.py`

## How to start the Automated tests (Pytest)
To start the automated tests, simply navigate to the root directory of the project and tun the command `pytest`. This will run tests for all functions used as construction blocks for the app.
