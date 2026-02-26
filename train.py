from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle
import pandas as pd

# Load your data here
df = pd.read_csv("path/to/your/data.csv")
X = df.drop('target_column', axis=1)  # Replace 'target_column' with your target column name
y = df['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()

param_grid = {
    'n_estimators': [100,200],
    'max_depth': [None,10,20]
}

grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

pickle.dump(best_model, open("model/model.pkl", "wb"))