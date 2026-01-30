STOCK PRICE PREDICTION USING LSTM

Project Overview
This project implements a Long Short-Term Memory (LSTM) based deep learning model to predict stock closing prices using historical market data. The model is trained on past stock prices and generates both test predictions and future price forecasts.

Features
- LSTM-based deep learning architecture
- Historical stock price preprocessing and normalization
- Train-test split for model evaluation
- RMSE-based performance evaluation
- Future stock price forecasting
- Automatic generation of prediction graphs

Model Architecture
- 3 stacked LSTM layers
- Dense output layer
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

Project Structure
Stock-Price-Prediction-LSTM/
│
├── train_model.py        # LSTM model training and forecasting
├── README.md             # Project documentation
├── .gitignore            # Ignored files configuration
└── LICENSE               # MIT License


Results
- Trained multiple models using different epoch values
- Generated RMSE comparison for model evaluation
- Visualized actual vs predicted stock prices
- Forecasted future stock prices for a specified number of days

---

How to Run
1. Clone the repository
   git clone
2. Install required libraries
   pip install -r requirements.txt
3. Run the training script
   python train_model.py

Dataset
Historical stock market data containing date-wise closing prices.  
(Data source: Yahoo Finance / NSE – dataset to be placed locally before execution.)


Future Enhancements
- Web-based dashboard for stock visualization
- REST API for real-time predictions
- Support for multiple stocks (NIFTY 50)
- Hyperparameter tuning and model optimization
