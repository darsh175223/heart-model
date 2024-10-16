import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

# Load data with specified data types
dtypes = {
    'age': float,
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

# Feature engineering (same as before)
data['age_squared'] = data['age'] ** 2
data['creatinine_ratio'] = data['serum_creatinine'] / data['creatinine_phosphokinase']
data['sodium_platelets_ratio'] = data['serum_sodium'] / data['platelets']
data['health_score'] = data['ejection_fraction'] - data['serum_creatinine'] * 10
data['age_diabetes'] = data['age'] * data['diabetes']
data['hbp_smoking'] = data['high_blood_pressure'] * data['smoking']
data['log_creatinine_phosphokinase'] = np.log1p(data['creatinine_phosphokinase'])
data['log_platelets'] = np.log1p(data['platelets'])
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
data['ef_group'] = pd.cut(data['ejection_fraction'], bins=[0, 30, 50, 70, 100], labels=['Very Low', 'Low', 'Normal', 'High'])

# One-hot encode categorical variables
categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'age_group', 'ef_group']
data_encoded = pd.get_dummies(data, columns=categorical_features)

# Separate features and target
X = data_encoded.drop(['DEATH_EVENT', 'time'], axis=1)
y = data_encoded['DEATH_EVENT']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Optuna objective function for Random Forest
def objective_rf(trial):
    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=trial.suggest_int('poly_degree', 1, 3), include_bias=False)),
        ('selector', SelectFromModel(RandomForestClassifier(random_state=42))),
        ('rf', RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 5, 30),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            random_state=42
        ))
    ])
    
    score = cross_val_score(rf_pipeline, X_train, y_train, n_jobs=-1, cv=5, scoring='roc_auc')
    return score.mean()

# Optuna study for Random Forest
study_rf = optuna.create_study(direction='maximize')
study_rf.optimize(objective_rf, n_trials=100)

print("Best Random Forest parameters:", study_rf.best_params)
print("Best Random Forest AUC:", study_rf.best_value)

# Train final Random Forest model with best parameters
best_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=study_rf.best_params['poly_degree'], include_bias=False)),
    ('selector', SelectFromModel(RandomForestClassifier(random_state=42))),
    ('rf', RandomForestClassifier(**{k: v for k, v in study_rf.best_params.items() if k != 'poly_degree'}, random_state=42))
])
best_rf.fit(X_train, y_train)

rf_pred = best_rf.predict(X_test)
rf_pred_proba = best_rf.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
print(f"Random Forest AUC: {rf_auc:.3f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Optuna objective function for LightGBM
def objective_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0)
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Use cv instead of train
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=1000,
        nfold=5,
        stratified=True,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return cv_results['auc-mean'][-1]

# Optuna study for LightGBM
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=100)

print("Best LightGBM parameters:", study_lgb.best_params)
print("Best LightGBM AUC:", study_lgb.best_value)

# Train final LightGBM model with best parameters
best_params_lgb = study_lgb.best_params
best_params_lgb['objective'] = 'binary'
best_params_lgb['metric'] = 'auc'

lgb_train = lgb.Dataset(X_train, y_train)
final_lgb_model = lgb.train(best_params_lgb, lgb_train, num_boost_round=1000, 
                            valid_sets=[lgb_train], early_stopping_rounds=50)

# Make predictions
lgb_pred = final_lgb_model.predict(X_test)
lgb_pred_binary = [1 if p >= 0.5 else 0 for p in lgb_pred]
lgb_accuracy = accuracy_score(y_test, lgb_pred_binary)
lgb_auc = roc_auc_score(y_test, lgb_pred)
print(f"LightGBM Accuracy: {lgb_accuracy:.3f}")
print(f"LightGBM AUC: {lgb_auc:.3f}")
print("\nLightGBM Classification Report:")
print(classification_report(y_test, lgb_pred_binary))

# Feature importance (Random Forest)
rf_feature_imp = pd.DataFrame({'feature': X.columns, 'importance': best_rf.named_steps['rf'].feature_importances_})
rf_feature_imp = rf_feature_imp.sort_values('importance', ascending=False)
print("\nTop 5 important features (Random Forest):")
print(rf_feature_imp.head())

# SHAP values for LightGBM
explainer = shap.TreeExplainer(final_lgb_model)
shap_values = explainer.shap_values(X_test)

shap_sum = np.abs(shap_values[1]).mean(axis=0)
importance_df = pd.DataFrame({'column_name': X_test.columns, 'shap_importance': shap_sum})
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

# Visualizations
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

plt.figure(figsize=(12, 6))
lgb.plot_importance(final_lgb_model, max_num_features=10)
plt.title("LightGBM Feature Importance")
plt.tight_layout()
plt.savefig('lgbm_feature_importance.png')
plt.close()

# Correlation matrix of top features
top_10_features = importance_df['column_name'].head(10).tolist()
correlation_matrix = X[top_10_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title("Correlation Matrix of Top 10 Features")
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Plot optimization history
plt.figure(figsize=(12, 6))
optuna.visualization.plot_optimization_history(study_rf)
plt.title("Random Forest Optimization History")
plt.tight_layout()
plt.savefig('rf_optimization_history.png')
plt.close()

plt.figure(figsize=(12, 6))
optuna.visualization.plot_optimization_history(study_lgb)
plt.title("LightGBM Optimization History")
plt.tight_layout()
plt.savefig('lgb_optimization_history.png')
plt.close()