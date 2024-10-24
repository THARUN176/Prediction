import os
import pickle
import pandas as pd
from flask import Flask, render_template, request
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import timedelta
from scipy.stats import randint, uniform

app = Flask(__name__)

# List of existing datasets
datasets = {
    "Hudson_Valley": r"datasets/Hudson_Valley.csv",
    "North": r"datasets/North.csv",
    "NYC": r"datasets/NYC.csv",
    "West": r"datasets/West.csv",
}

pickles_folder = 'pickles'

# Load existing models from pickles
models = {}
for dataset in datasets:  
    pickle_file = os.path.join(pickles_folder, f"{dataset}.pkl")
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            models[dataset] = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    dataset_name = None
    accuracy_metrics = {}
    error_message = None
    actual_values = None
    predicted_values = None

    if request.method == 'POST':
        if 'dataset' in request.form:  # Existing dataset selected
            dataset_name = request.form['dataset']
            if dataset_name in models:
                try:
                    # Load the dataset
                    data = pd.read_csv(datasets[dataset_name])

                    # Prepare features and target variable
                    data['Date'] = pd.to_datetime(data['Date'])
                    data['Day_of_Month'] = data['Date'].dt.day
                    data['Day_of_Week'] = data['Date'].dt.dayofweek
                    data['Month'] = data['Date'].dt.month
                    data['Year'] = data['Date'].dt.year
                    data['Hour'] = data['Date'].dt.hour

                    X = data[['Day_of_Month', 'Day_of_Week', 'Month', 'Year', 'Hour']]
                    y = data['Load(kW)']

                    # Use the loaded model for prediction
                    model = models[dataset_name]
                    y_pred = model.predict(X)

                    # Create a list of actual and predicted values
                    actual_values = y.values.tolist()
                    predicted_values = y_pred.tolist()

                    # Prepare the input data for prediction
                    start_date_str = request.form['start_date']
                    end_date_str = request.form['end_date']
                    start_date = pd.to_datetime(start_date_str)
                    end_date = pd.to_datetime(end_date_str)

                    # Validate start and end dates
                    if start_date > end_date:
                        error_message = "End date should be always greater than start date."
                        return render_template('index.html', datasets=datasets, predictions=None,
                                               dataset_name=dataset_name, accuracy_metrics={},
                                               error_message=error_message)

                    # Generate hourly predictions between start_date and end_date
                    predictions = []
                    current_date = start_date
                    fudge_factor = float(request.form['fudge_factor']) / 100  # Convert to decimal
                    while current_date <= end_date:
                        input_data = pd.DataFrame({
                            'Day_of_Month': [current_date.day],
                            'Day_of_Week': [current_date.dayofweek],
                            'Month': [current_date.month],
                            'Year': [current_date.year],
                            'Hour': [current_date.hour]
                        })
                        prediction = round(model.predict(input_data)[0] * (1 + fudge_factor))  # Apply fudge factor
                        predictions.append((current_date.strftime('%Y-%m-%d %H:%M'), prediction))
                        current_date += timedelta(hours=1)

                    # Calculate accuracy metrics
                    r2 = round(r2_score(y, y_pred), 3)
                    avg_error_percentage = f"{round((100 * abs(y - y_pred) / y).mean(), 2)}%"
                    accuracy_metrics = {
                        'R-Squared (R²)': r2 if r2 >= 0 else 0,
                        'Average Error Percentage': avg_error_percentage
                    }
                except Exception as e:
                    error_message = f"Error processing dataset: {str(e)}"
            else:
                error_message = "Model not found for the selected dataset."

        elif 'file' in request.files:  # New dataset uploaded
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                try:
                    # Save the uploaded file
                    uploaded_file_path = os.path.join('uploads', file.filename)
                    file.save(uploaded_file_path)

                    # Load the new dataset
                    new_data = pd.read_csv(uploaded_file_path)

                    # Validate columns
                    # Validate columns
                    if not {'Date', 'Load(kW)'}.issubset(new_data.columns):
                        error_message = "The uploaded dataset must contain 'Date' and 'Load(kW)' columns."
                    else:
                        # Prepare features and target variable
                        new_data['Date'] = pd.to_datetime(new_data['Date'])
                        new_data['Day_of_Month'] = new_data['Date'].dt.day
                        new_data['Day_of_Week'] = new_data['Date'].dt.dayofweek
                        new_data['Month'] = new_data['Date'].dt.month
                        new_data['Year'] = new_data['Date'].dt.year
                        new_data['Hour'] = new_data['Date'].dt.hour
                        X = new_data[['Day_of_Month', 'Day_of_Week', 'Month', 'Year', 'Hour']]
                        y = new_data['Load(kW)']

                        # Split the data into training and test sets
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                        # Define hyperparameter tuning space
                        param_grid = {
                                    'learning_rate': uniform(0.01, 0.3),  # Uniform distribution between 0.01 and 0.3
                                    'max_depth': randint(3, 10),  # Discrete uniform distribution between 3 and 10
                                    'n_estimators': randint(100, 1000),  # Discrete uniform distribution between 100 and 1000
                                    'colsample_bytree': uniform(0.5, 0.5),  # Uniform distribution between 0.5 and 1.0
                                    'alpha': uniform(0, 10)  # Uniform distribution between 0 and 10
                                         }

                        # Perform hyperparameter tuning using RandomizedSearchCV
                        from xgboost import XGBRegressor
                        from sklearn.model_selection import RandomizedSearchCV
                        xgb_model = XGBRegressor()
                        randomized_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, cv=5, n_iter=50, n_jobs=-1)
                        randomized_search.fit(X_train, y_train)

                        # Get the best model and hyperparameters
                        best_model = randomized_search.best_estimator_
                        best_params = randomized_search.best_params_

                        # Save the trained model with a unique name
                        dataset_name = file.filename.split('.')[0]
                        pickle_file = os.path.join(pickles_folder, f"{dataset_name}_model.pkl")
                        with open(pickle_file, 'wb') as f:
                            pickle.dump(best_model, f)

                        # Update the models dictionary
                        models[dataset_name] = best_model

                        # Use the best model for prediction
                        y_pred = best_model.predict(X_test)

                        # Create a list of actual and predicted values
                        actual_values = y_test.values.tolist()
                        predicted_values = y_pred.tolist()

                        # Prepare the input data for prediction
                        start_date_str = request.form['start_date']
                        end_date_str = request.form['end_date']
                        start_date = pd.to_datetime(start_date_str)
                        end_date = pd.to_datetime(end_date_str)

                        # Validate start and end dates
                        if start_date > end_date:
                            error_message = "End date should be always greater than start date."
                            return render_template('index.html', datasets=datasets, predictions=None,
                                                   dataset_name=dataset_name, accuracy_metrics={},
                                                   error_message=error_message)

                        # Generate hourly predictions between start_date and end_date
                        predictions = []
                        current_date = start_date
                        fudge_factor = float(request.form['fudge_factor']) / 100  # Convert to decimal
                        while current_date <= end_date:
                            input_data = pd.DataFrame({
                                'Day_of_Month': [current_date.day],
                                'Day_of_Week': [current_date.dayofweek],
                                'Month': [current_date.month],
                                'Year': [current_date.year],
                                'Hour': [current_date.hour]
                            })
                            prediction = round(best_model.predict(input_data)[0] * (1 + fudge_factor))  # Apply fudge factor
                            predictions.append((current_date.strftime('%Y-%m-%d %H:%M'), prediction))
                            current_date += timedelta(hours=1)

                        # Calculate accuracy metrics
                        r2 = round(r2_score(y_test, y_pred), 2)
                        avg_error_percentage = f"{round((100 * abs(y_test - y_pred) / y_test).mean(), 2)}%"
                        accuracy_metrics = {
                            'R-Squared (R²)': r2 if r2 >= 0 else 0,
                            'Average Error Percentage': avg_error_percentage
                        }
                except Exception as e:
                    error_message = f"Error processing dataset: {str(e)}"
            else:
                error_message = "Invalid file uploaded."

    return render_template('index.html', datasets=datasets, predictions=predictions,
                           dataset_name=dataset_name, accuracy_metrics=accuracy_metrics,
                           error_message=error_message, actual_values=actual_values,
                           predicted_values=predicted_values)

if __name__ == '__main__':
    app.run(debug=True)