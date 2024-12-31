# -*- coding: gbk -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime


def load_and_prepare_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    # Convert the 'date' column to datetime type
    data['date'] = pd.to_datetime(data['date'])
    # Set 'date' as the index
    data.set_index('date', inplace=True)
    return data

def forecast_co2(data, output_path='prediction_results.csv'):
    # Prepare CO2 data
    co2_data = data['CO2'].values
    dates = data.index

    # Split into training and testing sets (70-30 split)
    train_size = int(len(co2_data) * 0.7)
    train = co2_data[:train_size]
    test = co2_data[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    # Use auto_arima to find the best parameters
    print("\nSearching for the best ARIMA parameters...")
    model = auto_arima(train,
                       start_p=0, start_q=0,
                       max_p=5, max_q=5,
                       m=1,
                       d=1,  # Specify the order of differencing
                       seasonal=False,
                       trace=True,
                       error_action='ignore',
                       suppress_warnings=True,
                       stepwise=True)

    print(f"\nBest model parameters: {model.order}")

    # Stepwise forecasting
    predictions = []
    history = train.tolist()

    for i in range(len(test)):
        # Perform one-step prediction with the current model
        pred = model.predict(n_periods=1)[0]
        predictions.append(pred)

        # Update the history with the actual test value
        history.append(test[i])
        # Refit the model every 5 steps
        if i % 5 == 0:
            model.fit(history)

    # Create a DataFrame for prediction results
    results_df = pd.DataFrame({
        'date': test_dates,
        'actual': test,
        'predicted': predictions
    })
    results_df.set_index('date', inplace=True)

    # Save the prediction results
    results_df.to_csv(output_path)

    # Calculate MAE and RMSE
    mae = mean_absolute_error(test, predictions)
    rmse = mean_squared_error(test, predictions, squared=False)
    print(f'\nMAE (Mean Absolute Error): {mae:.2f}')
    print(f'RMSE (Root Mean Squared Error): {rmse:.2f}')

    # Plot the prediction results
    plt.figure(figsize=(15, 7))
    plt.plot(train_dates, train, label='Training Data', color='blue')
    plt.plot(test_dates, test, label='Actual Values', color='red')
    plt.plot(test_dates, predictions, label='Predicted Values', color='green')
    plt.title('CO2 Concentration Prediction Results')
    plt.xlabel('Date')
    plt.ylabel('CO2 Concentration')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return results_df

def main():
    # Load the dataset
    data = load_and_prepare_data('dataset.csv')

    # Display basic information about the dataset
    print("Dataset Basic Information:")
    print(data.head())
    print("\nDescriptive Statistics of the Dataset:")
    print(data.describe())

    # Perform CO2 forecasting
    results = forecast_co2(data)

    # Display a summary of the prediction results
    print("\nPrediction Results Summary:")
    print(results.head())

    # Correlation analysis
    correlation_matrix = data[['Temperature', 'Humidity', 'Light', 'CO2']].corr()
    print("\nCorrelation Analysis:")
    print(correlation_matrix)

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Sensor Data Correlation Matrix")
    plt.show()

if __name__ == "__main__":
    main()
