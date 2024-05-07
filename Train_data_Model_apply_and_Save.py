import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('path of IPL_Auction_Data.csv')
df.head()

df.shape

df.info()

df = df.dropna()

df.isnull().sum().sum()

df['Year'] = df['Year'].astype(int)

df.head()

from sklearn.preprocessing import LabelEncoder

le_player = LabelEncoder()
le_role = LabelEncoder()
le_team = LabelEncoder()
le_origin = LabelEncoder()
df['Player'] = le_player.fit_transform(df['Player'])
df['Role'] = le_role.fit_transform(df['Role'])
df['Team'] = le_team.fit_transform(df['Team'])
df['Player Origin'] = le_origin.fit_transform(df['Player Origin'])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error
import numpy as np
train_rmse_lr = np.sqrt(mean_squared_error(y_train, lr_model.predict(x_train)))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, lr_model.predict(x_test)))
train_rmse_lr
test_rmse_lr

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train, y_train)
train_rmse_dt = np.sqrt(mean_squared_error(y_train, dt_model.predict(x_train)))
test_rmse_dt = np.sqrt(mean_squared_error(y_test, dt_model.predict(x_test)))
train_rmse_dt
test_rmse_dt

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(x_train, y_train)
train_rmse_rf = np.sqrt(mean_squared_error(y_train, rf_model.predict(x_train)))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, rf_model.predict(x_test)))
train_rmse_rf
test_rmse_rf

from sklearn.svm import SVR
svr_model = SVR(kernel='rbf')
svr_model.fit(x_train, y_train)
train_rmse_svr = np.sqrt(mean_squared_error(y_train, svr_model.predict(x_train)))
test_rmse_svr = np.sqrt(mean_squared_error(y_test, svr_model.predict(x_test)))
train_rmse_svr
test_rmse_svr

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
degree = 2  # You can change the degree as needed
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_model.fit(x_train, y_train)
train_rmse_poly = np.sqrt(mean_squared_error(y_train, poly_model.predict(x_train)))
test_rmse_poly = np.sqrt(mean_squared_error(y_test, poly_model.predict(x_test)))
train_rmse_poly
test_rmse_poly

from sklearn.ensemble import GradientBoostingRegressor
gb_model = GradientBoostingRegressor()
gb_model.fit(x_train, y_train)
train_rmse_gb = np.sqrt(mean_squared_error(y_train, gb_model.predict(x_train)))
test_rmse_gb = np.sqrt(mean_squared_error(y_test, gb_model.predict(x_test)))
train_rmse_gb
test_rmse_gb


rmse_data = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'SVR', 'Polynomial Regression', 'Gradient Boosting'],
    'Train RMSE': [train_rmse_lr, train_rmse_dt, train_rmse_rf, train_rmse_svr, train_rmse_poly, train_rmse_gb],
    'Test RMSE': [test_rmse_lr, test_rmse_dt, test_rmse_rf, test_rmse_svr, test_rmse_poly, test_rmse_gb]
}
rmse_df = pd.DataFrame(rmse_data)
print(rmse_df)

# Based on the RMSE values I got, the Gradient Boosting model appears to be the best model, as it has the lowest RMSE on both training and testing data, it is also not overfitted.

import pickle
pickle.dump(gb_model, open('./model_new1.sav', 'wb'))


