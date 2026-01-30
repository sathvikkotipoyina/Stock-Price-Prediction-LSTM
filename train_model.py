import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# --------------------------------
# Helper: Create Dataset
# --------------------------------
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


# --------------------------------
# Main Function: Train and Save Models
# --------------------------------
def train_and_save_models(file_path, future_days=60, time_step=100,
                          epoch_list=[5, 6, 7, 8], stock_name="SBIN"):

    # Setup folder
    result_dir = f"results_{future_days}_{stock_name}"
    os.makedirs(result_dir, exist_ok=True)

    # Load dataset
    print(f"📂 Loading dataset: {file_path}")
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df1 = df["Close"].astype(float).values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1_scaled = scaler.fit_transform(df1)

    # Split train/test
    training_size = int(len(df1_scaled) * 0.65)
    train_data = df1_scaled[:training_size]
    test_data = df1_scaled[training_size:]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Summary storage
    summary = []

    # Loop through epoch values
    for EPOCHS in epoch_list:
        print(f"\n🚀 Training model for {EPOCHS} epochs...")

        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50, return_sequences=True),
            LSTM(50),
            Dense(1)
        ])
        model.compile(loss="mean_squared_error", optimizer="adam")

        # Train
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=EPOCHS, batch_size=64, verbose=0)

        # Predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        train_rmse = math.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = math.sqrt(mean_squared_error(y_test_actual, test_predict))
        summary.append({"Epochs": EPOCHS, "Train_RMSE": train_rmse, "Test_RMSE": test_rmse})

        print(f"✅ Epoch {EPOCHS}: Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}")

        # Save Model
        model.save(os.path.join(result_dir, f"model_epochs_{EPOCHS}.h5"))

        # ---------------------------
        # Test Prediction Graph
        # ---------------------------
        look_back = time_step
        trainPlot = np.empty_like(df1_scaled)
        trainPlot[:, :] = np.nan
        trainPlot[look_back:len(train_predict) + look_back, :] = train_predict

        testPlot = np.empty_like(df1_scaled)
        testPlot[:, :] = np.nan
        testPlot[len(train_predict) + (look_back * 2) + 1:len(df1_scaled) - 1, :] = test_predict

        plt.figure(figsize=(12, 6))
        plt.plot(scaler.inverse_transform(df1_scaled), label="Actual Price", color='gray')
        plt.plot(trainPlot, label="Train Prediction", color='blue')
        plt.plot(testPlot, label="Test Prediction", color='red')
        plt.title(f"{stock_name} - Test Prediction ({EPOCHS} epochs)")
        plt.xlabel("Days")
        plt.ylabel("Close Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"test_graph_epochs_{EPOCHS}.png"))
        plt.close()

        # ---------------------------
        # Future Forecast Graph
        # ---------------------------
        temp_input = list(df1_scaled[-time_step:].reshape(-1))
        lst_output = []

        for i in range(future_days):
            x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

        future_prices = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        future_plot = np.arange(len(df1), len(df1) + future_days)
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(len(df1)), scaler.inverse_transform(df1_scaled), label="Historical", color='gray')
        plt.plot(future_plot, future_prices, label=f"Next {future_days} Days", color='orange')
        plt.title(f"{stock_name} - Future Forecast ({EPOCHS} epochs)")
        plt.xlabel("Days")
        plt.ylabel("Predicted Close Price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"forecast_graph_epochs_{EPOCHS}.png"))
        plt.close()

    # Save RMSE summary
    pd.DataFrame(summary).to_csv(os.path.join(result_dir, "rmse_summary.csv"), index=False)
    print(f"\n📁 All models, graphs, and results saved in: {result_dir}")


# Example Usage
train_and_save_models("SBIN.NS_daily_technical_merged.csv",
                      future_days=120,
                      time_step=100,
                      epoch_list=[ 7, 8, 9],
                      stock_name="TCS")
