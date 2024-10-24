
# Load Forecasting Application

## Overview

The Load Forecasting Application is a tool designed to predict energy load based on historical data. It allows users to either select predefined datasets or upload their own datasets for prediction. The application uses machine learning models, specifically XGBoost, for forecasting and provides flexibility to apply fudge factors for tuning predictions.

## Features

- **Select Predefined Dataset**: Users can select an existing dataset from a predefined list for load predictions.
- **Upload Custom Dataset**: Users can upload their own datasets following the specified schema.
- **Date Selection**: The user can select a date range for which the predictions will be made.
- **Fudge Factor(Optional)**: Users can apply a fudge factor if they wish.
- **Prediction Display**: The app displays the predicted values along with accuracy metrics (R² and Average Error Percentage).
- **Download Sample Dataset**: Users can download a sample CSV file that adheres to the required format for dataset uploads.
- **Print Option**: The results and predictions can be printed directly from the interface.

---

## Project Structure

```
├── app.py
├── static
│   ├── style.css
│   ├── download.png
│   ├── sample.csv
└── templates
    └── index.html
├── README.md
```

---

## Requirements

- Python 3.7 or higher
- Flask
- XGBoost
- Pandas
- Scikit-learn

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

---

## Running the Application

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask application:**

    ```bash
    python app.py
    ```

4. **Access the application**:  
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Application Details

### 1. HTML Page (`index.html`)

This is the main interface of the application where users can interact with the load forecasting model.

- **Header**: Displays the TMEIC logo and a welcome message.
- **Option Selection**: Provides buttons to select either a predefined dataset or upload a custom dataset.
- **Forms**: Depending on the user’s choice, a form appears to either select from the existing datasets or upload a new one. It includes date selection, fudge factor, and submission button.
- **Results**: After making predictions, the page displays a table with predicted load values and relevant accuracy metrics.
- **Print Button**: Allows users to print the prediction results directly from the page.

### 2. CSS Stylesheet (`style.css`)

This file handles the design and layout of the HTML page.

- **Container**: Centers and styles the main content block with padding, shadows, and rounded corners.
- **Buttons and Forms**: Styles buttons, input fields, tables, and labels for a clean, user-friendly interface.
- **Responsive Design**: Ensures the layout adjusts for smaller screens, making the app accessible on mobile devices.

### 3. Flask Application (`app.py`)

This is the backend script that powers the application.

- **Routes**:  
  - `/` : Renders the home page where users can select a dataset or upload a custom one.
  - `/predict` : Handles the prediction logic using the selected or uploaded dataset, applying the specified date range and fudge factor.

- **Model Loading**: Pretrained XGBoost models are loaded to perform predictions.
- **Dataset Handling**: The app handles both predefined datasets (stored in the `static` folder) and new uploads. Uploaded files are validated based on schema before processing.
- **Prediction**: Predictions are generated based on the date range and fudge factor, and results are displayed with R² and Average Error Percentage.

---

## Dataset Guidelines

When uploading a new dataset, please ensure it follows the below format:

| Column Name | Data Type | Description |
|-------------|-----------|-------------|
| Date        | datetime  | Date and time of the load measurement (e.g., 1/1/2016 12:00:00 AM) |
| Load        | float     | Measured load in kilowatts (kW) |

You can download a sample dataset from the application to understand the structure required.

---

## How to Use

1. **Select Existing Dataset**:
   - Click the "Select Existing Dataset" button.
   - Choose a predefined dataset from the dropdown.
   - Select a start and end date.
   - Choose a fudge factor (if needed).
   - Click the "Predict" button to generate predictions.

2. **Upload Custom Dataset**:
   - Click the "Upload Your Own Dataset" button.
   - Upload a CSV file following the required schema.
   - Select a start and end date.
   - Choose a fudge factor (if needed).
   - Click the "Predict" button to generate predictions.

3. **View Results**:  
   The predicted load values will be displayed in a table format along with the accuracy metrics. You can print the results directly using the "Print Results" button.

---

## Troubleshooting

- **Invalid File Upload**: Ensure that the uploaded dataset adheres to the required schema (Date and Load columns).
- **Error Message**: If there are any issues, the error message will be displayed in red at the bottom of the page. Review the message to debug the problem.

---

## Future Improvements

- Add support for additional machine learning models.
- Implement time-series visualization of predicted vs. actual loads.
- Improve the accuracy metrics by incorporating additional performance measures like MAE, RMSE.

---
