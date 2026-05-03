# House Price Prediction
An end-to-end Machine Learning web application that predicts California house prices based on demographic and property data. 

This project features a trained Random Forest regression model wrapped in a modern, interactive "glass-morphism" web UI, complete with dynamic mapping and interactive sensitivity analysis.

## ✨ Features
* **Machine Learning Model:** Utilizes a `RandomForestRegressor` trained on the California Housing dataset.
* **Interactive UI:** Built with Streamlit, featuring a custom CSS background and glass-morphism data cards.
* **Geospatial Mapping:** Automatically generates an interactive map plotting the target property based on latitude/longitude inputs.
* **Sensitivity Analysis:** Integrates Plotly to generate a dynamic "Price Trend" chart, showing how the predicted value scales with neighborhood median income while holding other features constant.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Frontend:** Streamlit
* **Data Visualization:** Plotly Express

## 🚀 How to Run Locally

**Note:** The `house_model.pkl` file is not included in this repository due to GitHub file size limits. You must generate it locally before running the web app.

**1. Clone the repository**
```bash
git clone [https://github.com/atulkumar2025/house-price-prediction.git](https://github.com/atulkumar2025/house-price-prediction.git)
cd house-price-prediction
2. Install dependencies

Bash
pip install -r requirements.txt
3. Generate the Machine Learning Model
Open model.ipynb in your preferred Jupyter environment (VS Code, JupyterLab, etc.) and run all the cells. This will download the dataset, train the Random Forest model, and generate the required house_model.pkl file in your directory.

4. Run the Streamlit App

Bash
streamlit run app.py
🧠 Model Architecture Overview
Algorithm: Random Forest Regressor

Preprocessing: StandardScaler applied to normalize feature weights (e.g., balancing high-value Population data against lower-value Bedroom counts).

Evaluation Metrics: Evaluated using Root Mean Squared Error (RMSE) and R-squared score to ensure generalization against unseen test data.
