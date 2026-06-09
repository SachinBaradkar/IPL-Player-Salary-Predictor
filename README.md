🏏 IPL Player Salary Predictor

Predict the auction salary of an IPL player using Machine Learning.

📌 Overview

IPL Player Salary Predictor is a Streamlit-based Machine Learning web application that estimates the auction salary of an IPL player based on historical IPL auction data.

The application uses a trained Gradient Boosting Regression model to analyze player-related features such as role, team, origin, and auction year to predict the player's expected salary.

This project demonstrates the practical application of Machine Learning in sports analytics, specifically in player valuation and auction price prediction.

🚀 Features

✅ Interactive Streamlit Web Application

✅ IPL Player Search Functionality

✅ Team Selection Dropdown

✅ Player Role Selection

✅ Indian/Overseas Player Classification

✅ Future Year Prediction Support (2025–2027)

✅ Instant Salary Prediction

✅ Machine Learning Powered Decision Making

📸 Application Preview

The application provides:

Player Name Input
Role Selection
Team Selection
Year Selection
Player Origin Selection
Predicted Salary Output
🛠️ Tech Stack
Frontend
Streamlit
Backend
Python
Machine Learning
Scikit-Learn
Gradient Boosting Regressor
Data Processing
Pandas
NumPy
Data Storage
CSV
Excel Files
📂 Project Structure
IPL-Player-Salary-Predictor/
│
├── app.py
├── Train_data_Model_apply_and_Save.py
├── IPL_Auction_Data.csv
├── model_new1.sav
│
├── player_name.xlsx
├── team_name.xlsx
├── role.xlsx
├── origin.xlsx
│
├── iplaml.jpg
├── requirements.txt
└── README.md
📊 Dataset

The project uses historical IPL auction data containing:

Feature	Description
Player	Player Name
Role	Batsman, Bowler, All-Rounder, Wicket Keeper
Team	IPL Franchise
Year	Auction Year
Player Origin	Indian / Overseas
Salary	Auction Price
🤖 Machine Learning Models Evaluated

Several regression algorithms were trained and compared:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Support Vector Regression (SVR)
Polynomial Regression
Gradient Boosting Regressor
Final Model Selected

🏆 Gradient Boosting Regressor

Reason:

Lowest RMSE
Better generalization
Reduced overfitting

The trained model is stored as:

model_new1.sav
🔍 Machine Learning Workflow
1. Data Collection

Historical IPL auction dataset was collected.

2. Data Cleaning
Missing values removed
Datatypes corrected
3. Feature Encoding

Categorical columns converted using Label Encoding:

Player
Role
Team
Player Origin
4. Train-Test Split
test_size = 0.2
80% Training
20% Testing
5. Model Training

Multiple regression algorithms were trained and compared.

6. Model Evaluation

Evaluation Metric:

RMSE (Root Mean Square Error)
7. Model Deployment

The best model was saved using Pickle and deployed through Streamlit.

⚙️ Installation
Clone Repository
git clone https://github.com/SachinBaradkar/IPL-Player-Salary-Predictor.git
cd IPL-Player-Salary-Predictor
Install Dependencies
pip install -r requirements.txt
▶️ Run Application
streamlit run app.py

The application will open in your browser automatically.

🧠 How Prediction Works
User enters player name.
Application retrieves encoded player ID.
User selects:
Role
Team
Year
Origin
Features are passed to the trained Gradient Boosting model.
Model predicts expected IPL salary.
Salary is displayed in Indian Rupees (₹).
📈 Future Improvements
Add player performance statistics
Include batting and bowling metrics
Use XGBoost and CatBoost models
Live IPL database integration
Interactive salary trend visualization
Auction price comparison dashboard
Deep Learning based prediction
🎯 Applications
Sports Analytics
Cricket Data Science
Auction Price Forecasting
Team Management Strategy
Player Valuation Analysis
