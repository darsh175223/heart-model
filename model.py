import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap

# Load data with specified data types
dtypes = {
    'anaemia': int,
    'creatinine_phosphokinase': int,
    'diabetes': int,
    'ejection_fraction': int,
    'high_blood_pressure': int,
    'platelets': float,
    'serum_creatinine': float,
    'serum_sodium': int,
    'sex': int,
    'smoking': int,
    'time': int,
    'DEATH_EVENT': int
}

data = pd.read_csv('heart_failure_clinical_records_dataset (1).csv', dtype=dtypes)

# Separate features and target
X = data.drop(['DEATH_EVENT', 'age', 'time', 'smoking'], axis=1)
y = data['DEATH_EVENT']

# Define numerical and categorical features
numerical_features = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Random Forest with hyperparameter optimization
rf = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
print("Best Random Forest parameters:", rf_grid.best_params_)

# LightGBM with hyperparameter optimization
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Define the parameter grid
param_grid = {
    'num_leaves': [31, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

# Perform grid search
best_params = {}
best_score = float('inf')

for num_leaves in param_grid['num_leaves']:
    for learning_rate in param_grid['learning_rate']:
        for n_estimators in param_grid['n_estimators']:
            params = lgb_params.copy()
            params['num_leaves'] = num_leaves
            params['learning_rate'] = learning_rate
            
            model = lgb.train(params, lgb_train, num_boost_round=n_estimators, 
                              valid_sets=[lgb_test], 
                              callbacks=[lgb.early_stopping(stopping_rounds=10)])
            
            score = model.best_score['valid_0']['binary_logloss']
            
            if score < best_score:
                best_score = score
                best_params = {'num_leaves': num_leaves, 'learning_rate': learning_rate, 'n_estimators': n_estimators}

print("Best LightGBM parameters:", best_params)

# Train final LightGBM model with best parameters
final_lgb_model = lgb.train({**lgb_params, **best_params}, lgb_train, num_boost_round=best_params['n_estimators'])

# Make predictions
lgb_pred = final_lgb_model.predict(X_test)
lgb_pred_binary = [1 if p >= 0.5 else 0 for p in lgb_pred]
lgb_accuracy = accuracy_score(y_test, lgb_pred_binary)
print(f"LightGBM Accuracy: {lgb_accuracy:.3f}")

# Feature importance (Random Forest)
importances = rf_best.feature_importances_
feature_imp = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_imp = feature_imp.sort_values('importance', ascending=False)
print("\nTop 5 important features (Random Forest):")
print(feature_imp.head())

# SHAP values for LightGBM
explainer = shap.TreeExplainer(final_lgb_model)
shap_values = explainer.shap_values(X_test)
shap_sum = np.abs(shap_values[0]).mean(axis=0)
importance_df = pd.DataFrame([X_test.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
print("\nTop 5 important features (SHAP):")
print(importance_df.head())

# Calculate contribution percentages for top two features
top_features = importance_df['column_name'].head(2).tolist()
total_importance = importance_df['shap_importance'].sum()
contributions = importance_df[importance_df['column_name'].isin(top_features)]
contributions['percentage'] = contributions['shap_importance'] / total_importance * 100

print("\nTop two feature contributions:")
for _, row in contributions.iterrows():
    print(f"{row['column_name']}: {row['percentage']:.1f}%")