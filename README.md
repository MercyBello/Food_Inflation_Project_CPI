# UK Food Inflation Forecasting Dashboard — Project (CPI MoM %)

## Project Outline

I built this project to project transforms official **UK Consumer Price Index (CPI)** data for *Food & Non-Alcoholic Beverages* into a fully operational data science solution — taking raw government statistics and turning them into interactive, decision-ready forecasts.

My project demonstrates a **complete analytics workflow** used in real-world data science projects

-  Data cleaning and transformation  
-  Exploratory Data Analysis (EDA)  
-  Time-series modelling (**Naïve**, **Seasonal-Naïve**, **SARIMA**, **Prophet**)  
-  Model evaluation with industry-standard metrics (**MAPE**, **RMSE**)  
-  Deployment preparation using **Streamlit** for interactive visual insights  

Built as a **personal portfolio project aimed at professional employment**, this solution simulates real-world data science tasks in **product analytics**, **consulting**, and **business intelligence** — helping organisations anticipate **cost-of-living trends** and communicate complex prediction results to non-technical stakeholders.

---

##  Key Skills & Tools Demonstrated

- **Data Wrangling:** Cleaning and transforming official ONS statistics into a tidy, analysis-ready time-series dataset  
- **Exploratory Data Analysis (EDA):** Visualising trends, volatility, and seasonality to uncover actionable insights  
- **Predictive Models:** Building and benchmarking classical time-series models — Naïve, Seasonal-Naïve, SARIMA, Prophet  
- **Model Evaluation:** Measuring performance with MAPE and RMSE and applying promotion rules for model selection  
- **Deployment-Ready App:** Building an interactive **Streamlit** dashboard for clear, stakeholder-friendly visualisation  
- **Responsible Communication:** Reporting model uncertainty, explaining limitations, and ensuring transparent forecasting decisions

---

##  Why This Project Matters

I worked this project to demonstrates the **complete lifecycle of a real-world data science solution** — from sourcing and transforming raw official statistics into structured, analysis-ready datasets, to delivering exploratory insights, building and benchmarking forecasting models, and deploying interactive, stakeholder-facing visualisations.

My work reflects the same **rigorous workflows used by data science and AI teams**, other global technology leaders, as well as in **fintech, consulting, and public sector organisations**.

By combining **technical depth** (data engineering, EDA, forecasting with SARIMA and Prophet, model evaluation) with **business impact** (predicting cost-of-living trends, communicating uncertainty, and informing strategic decisions), this project illustrates the ability to deliver **scalable, production-ready analytics solutions** from start to finish.

---

##  Tech Stack

- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, Statsmodels, Prophet  
- **Deployment:** Streamlit
- **Data Source:** Office for National Statistics (ONS) – UK Consumer Price Index (CPI), Food & Non-Alcoholic Beverages (MoM %)

---

## How to Run

1. Clone this repository (Git Bash was used):

```bash
git clone https://github.com/your-username/uk-food-inflation-forecasting.git
cd uk-food-inflation-forecasting
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv

.venv\Scripts\activate (For Windows)
source .venv/bin/activate (For macOS/Linux)
```

3. Create a txt folder in your file directory (requirements.txt)
paste the following in an ordinary list format in the txt file (requirements.txt)
- pandas
- numpy
- matplotlib
- prophet
- scikit-learn
- streamlit

Run this code line to install all requirements
```bash
pip install -r requirements.txt
```

4. Use your preferred notebook

   
5. Launch the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```
