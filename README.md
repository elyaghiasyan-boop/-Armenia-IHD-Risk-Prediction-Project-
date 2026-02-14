

pip install numpy pandas scipy matplotlib seaborn scikit-learn xgboost shap joblib

# ==============================
# Step 0: Imports
# ==============================
import pandas as pd
import numpy as np
from scipy.stats import gamma, beta, expon, norm, vonmises, lognorm, pareto

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

sns.set(style='whitegrid')
logging.basicConfig(level=logging.INFO)

# ==============================
# Step 1: Load Data
# ==============================
# Health, environmental, and lifestyle data for Armenia
data = {
    'COVID19_deaths': [244.8, 200.5, 300.2, 250.0, 260.0],
    'Stroke_deaths': [85.9, 90.2, 80.3, 87.0, 83.5],
    'Trachea_lung_cancer_deaths': [42.6, 50.1, 40.5, 45.0, 44.0],
    'Lower_respiratory_infections': [34.7, 30.2, 35.0, 32.5, 31.0],
    'Hypertension_prevalence': [47.3, 45.2, 50.1, 48.0, 46.5],
    'Tobacco_use': [24.9, 22.1, 26.0, 25.0, 23.5],
    'Alcohol_consumption': [4.7, 5.1, 4.2, 4.8, 5.0],
    'Adult_obesity': [24.5, 22.0, 26.0, 25.0, 23.0],
    'Child_adolescent_obesity': [9.6, 8.2, 10.1, 9.5, 9.0],
    'Intimate_partner_violence_12mo': [5, 4, 6, 5, 4],
    'Intimate_partner_violence_lifetime': [10, 8, 12, 10, 9],
    'Wasting_under5': [4.4, 3.5, 5.0, 4.0, 3.8],
    'Safely_managed_drinking_water': [82, 85, 80, 83, 84],
    'Safely_managed_sanitation': [11, 12, 10, 11, 12],
    'Handwashing_facilities': [94, 92, 95, 93, 94],
    'Safely_treated_wastewater': [1, 2, 1, 1, 2],
    'Clean_fuels': [98.7, 97.0, 99.0, 98.5, 98.0],
    'PM2_5': [34.13, 35.5, 30.2, 33.0, 32.0],
    'Ischaemic_heart_disease_deaths': [378.9, 350.0, 400.5, 370.0, 365.0]
}
df = pd.DataFrame(data)
n = df.shape[0]

logging.info("Data loaded successfully")
print(df.head())

# ==============================
# Step 2: Synthetic Feature Generation
# ==============================
# Purpose: capture complex statistical variation
np.random.seed(42)

df['gamma_feature'] = gamma.rvs(a=2.0, size=n)
df['beta_feature'] = beta.rvs(a=2.0, b=5.0, size=n)
df['expon_feature'] = expon.rvs(scale=1.0, size=n)
df['gauss_feature'] = norm.rvs(loc=0.0, scale=1.0, size=n)
df['vonmises_feature'] = vonmises.rvs(kappa=1.0, size=n)
df['lognorm_feature'] = lognorm.rvs(s=0.5, scale=np.exp(1), size=n)
df['pareto_feature'] = pareto.rvs(b=2.0, size=n)

logging.info("Synthetic statistical features created")

# ==============================
# Step 3: Feature Engineering
# ==============================
# Combining variables, ratios, differences for richer representation
df['Hypertension_Obesity'] = df['Hypertension_prevalence'] * df['Adult_obesity']
df['PM2_5_per_clean_fuels'] = df['PM2_5'] / (df['Clean_fuels'] + 0.1)
df['Tobacco_Alcohol_ratio'] = df['Tobacco_use'] / (df['Alcohol_consumption'] + 0.1)
df['ChildAdult_Obesity_diff'] = df['Adult_obesity'] - df['Child_adolescent_obesity']

logging.info("Feature engineering completed")

# ==============================
# Step 4: Split Features & Target
# ==============================
X = df.drop('Ischaemic_heart_disease_deaths', axis=1)
y = df['Ischaemic_heart_disease_deaths']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
logging.info("Train-test split completed")

# ==============================
# Step 5: Preprocessing Pipeline
# ==============================
preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),       # fill missing values
    ('scaler', StandardScaler()),                        # standardize numeric features
    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # nonlinear interactions
    ('power', PowerTransformer(method='yeo-johnson'))   # stabilize variance
])
logging.info("Preprocessing pipeline created")

# ==============================
# Step 6: XGBoost Model Setup
# ==============================
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=1.0,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=2,
    early_stopping_rounds=50,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb_model)
])
logging.info("XGBoost pipeline setup completed")

# ==============================
# Step 7: Hyperparameter Tuning with GridSearchCV
# ==============================
param_grid = {
    'regressor__max_depth': [4,5,6],
    'regressor__learning_rate': [0.01,0.03,0.05],
    'regressor__n_estimators': [300,500,700],
    'regressor__gamma': [0,0.5,1.0],
    'regressor__subsample':[0.7,0.8,0.9],
    'regressor__colsample_bytree':[0.7,0.8,0.9]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='r2', verbose=1
)
grid_search.fit(X_train, y_train)
logging.info(f"Best hyperparameters: {grid_search.best_params_}")

# ==============================
# Step 8: Model Evaluation
# ==============================
y_pred = grid_search.predict(X_test)
logging.info(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
logging.info(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
logging.info(f"R²: {r2_score(y_test, y_pred):.2f}")

# ==============================
# Step 9: Feature Importance
# ==============================
best_model = grid_search.best_estimator_.named_steps['regressor']
plt.figure(figsize=(12,6))
xgb.plot_importance(best_model, importance_type='gain', max_num_features=20)
plt.title("XGBoost Feature Importance")
plt.show()

# ==============================
# Step 10: SHAP Explainable AI
# ==============================
explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type='bar')
# Detailed individual contributions
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_test.iloc[0,:])

# ==============================
# Step 11: Cross-Validation
# ==============================
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
scores = cross_val_score(grid_search.best_estimator_, X, y, scoring='r2', cv=cv)
logging.info(f"Cross-validated R² mean: {scores.mean():.3f}, std: {scores.std():.3f}")

 

