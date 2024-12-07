import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from category_encoders import TargetEncoder

# Function to optimize data types
def optimize_types(df):
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float' if df[col].dtype == 'float64' else 'integer')
    return df

# Load training data
print("Loading training data...")
train = pd.read_csv('train.csv')

# Handle missing values in geospatial columns
print("Handling missing values...")
train = train.dropna(subset=['lat', 'long', 'merch_lat', 'merch_long'])

# Calculate geospatial distance
print("Calculating geospatial distances...")
train['distance'] = train.apply(
    lambda x: geodesic((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km, axis=1
)

# Transform dates into useful features
print("Transforming dates...")
train['trans_date'] = pd.to_datetime(train['trans_date'])
train['trans_month'] = train['trans_date'].dt.month
train['trans_day'] = train['trans_date'].dt.day
train['trans_weekday'] = train['trans_date'].dt.weekday
train['age'] = pd.to_datetime('today').year - pd.to_datetime(train['dob']).dt.year

# Add new features
print("Adding new features...")
train['amt_per_city_pop'] = train['amt'] / train['city_pop']
train['is_weekend'] = train['trans_weekday'].isin([5, 6]).astype(int)
train['amt_distance_interaction'] = train['amt'] * train['distance']
train['amt_weekday_interaction'] = train['amt'] * train['trans_weekday']
train['log_amt'] = np.log1p(train['amt'])
train['transaction_speed'] = train['amt'] / (train['distance'] + 1)
train['amt_lat_long_interaction'] = train['amt'] * train['lat'] * train['long']

# Clustering
print("Adding cluster-based features...")
kmeans = KMeans(n_clusters=5, random_state=42)
train['cluster'] = kmeans.fit_predict(train[['amt', 'distance']])

# Fraud ratios
print("Calculating fraud ratios...")
fraud_ratio = train.groupby('category')['is_fraud'].mean()
train['category_fraud_ratio'] = train['category'].map(fraud_ratio)

# Drop unnecessary columns
columns_to_drop = ['trans_num', 'trans_date', 'trans_time', 'dob', 'first', 'last', 'street', 'cc_num', 'merchant']
train = train.drop(columns=columns_to_drop)

# Target encode categorical features
print("Target encoding categorical features...")
categorical_columns = ['category', 'state', 'gender', 'city', 'job']
encoder = TargetEncoder()
for col in categorical_columns:
    train[col] = encoder.fit_transform(train[col], train['is_fraud'])

# Optimize data types
train = optimize_types(train)

# Prepare training data
X = train.drop(columns=['is_fraud'])
y = train['is_fraud']

# Split data into training and validation sets
print("Handling class imbalance with RandomUnderSampler...")
undersampler = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = undersampler.fit_resample(X, y)

# Use StratifiedKFold for cross-validation
print("Using StratifiedKFold...")
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# Train XGBoost with RandomizedSearchCV
print("Training XGBoost with RandomizedSearchCV...")
param_grid = {
    'n_estimators': [200, 250],
    'max_depth': [8, 10],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.9],
    'colsample_bytree': [0.8]
}

random_search = RandomizedSearchCV(
    estimator=XGBClassifier(scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1]), random_state=42),
    param_distributions=param_grid,
    n_iter=5,
    scoring='f1',
    cv=2,
    verbose=1
)

random_search.fit(X_balanced, y_balanced)
best_xgb = random_search.best_estimator_
print(f"Best XGBoost Parameters: {random_search.best_params_}")

# Train Voting Classifier
print("Training Voting Classifier...")
voting_model = VotingClassifier(
    estimators=[
        ('xgb', best_xgb),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ],
    voting='soft',
    weights=[3, 2, 1]
)
voting_model.fit(X_balanced, y_balanced)

# Evaluate the model using cross-validation
for train_index, val_index in skf.split(X_balanced, y_balanced):
    X_train_fold, X_val_fold = X_balanced.iloc[train_index], X_balanced.iloc[val_index]
    y_train_fold, y_val_fold = y_balanced.iloc[train_index], y_balanced.iloc[val_index]
    voting_model.fit(X_train_fold, y_train_fold)
    y_pred_fold = voting_model.predict(X_val_fold)
    print(classification_report(y_val_fold, y_pred_fold))

# Load test data
print("Loading test data...")
test = pd.read_csv('test.csv')
test = test.dropna(subset=['lat', 'long', 'merch_lat', 'merch_long'])

# Apply the same feature engineering to test data
print("Transforming test data...")
test['distance'] = test.apply(lambda x: geodesic((x['lat'], x['long']), (x['merch_lat'], x['merch_long'])).km, axis=1)
test['trans_date'] = pd.to_datetime(test['trans_date'])
test['trans_month'] = test['trans_date'].dt.month
test['trans_day'] = test['trans_date'].dt.day
test['trans_weekday'] = test['trans_date'].dt.weekday
test['age'] = pd.to_datetime('today').year - pd.to_datetime(test['dob']).dt.year
test['amt_per_city_pop'] = test['amt'] / test['city_pop']
test['is_weekend'] = test['trans_weekday'].isin([5, 6]).astype(int)
test['amt_distance_interaction'] = test['amt'] * test['distance']
test['amt_weekday_interaction'] = test['amt'] * test['trans_weekday']
test['log_amt'] = np.log1p(test['amt'])
test['transaction_speed'] = test['amt'] / (test['distance'] + 1)
test['amt_lat_long_interaction'] = test['amt'] * test['lat'] * test['long']
test['cluster'] = kmeans.predict(test[['amt', 'distance']])
test['category_fraud_ratio'] = test['category'].map(fraud_ratio)

# Ensure all categorical columns exist in the test dataset
print("Ensuring all categorical columns exist in the test dataset...")
for col in categorical_columns:
    if col not in test.columns:
        print(f"Adding missing column '{col}' to test data with default value 'unknown'...")
        test[col] = "unknown"
    elif test[col].isnull().all():
        print(f"Column '{col}' exists but is entirely NaN, filling with 'unknown'...")
        test[col].fillna("unknown", inplace=True)

# Apply target encoding to categorical columns in test data
print("Applying target encoding to test data...")
for col in categorical_columns:
    test[col] = encoder.transform(test[col])

# Make predictions on the test data
print("Making predictions...")
test['is_fraud'] = voting_model.predict(test)

# Prepare submission file
print("Creating submission file...")
submission = test[['id']].copy()
submission['is_fraud'] = test['is_fraud']
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
