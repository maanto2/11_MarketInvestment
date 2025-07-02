import yfinance as yf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import smtplib
import datetime as dt
import threading
import time
import csv


# ---------------- CONFIG -------------------

# --------------- GLOBAL STORE --------------
data_store = {
    "actual": [],
    "predicted": [],
    "timestamps": [],
    "sentiment_score": []
}

# --------------- DATA FUNCTIONS ------------
def get_sp500_data(interval='5m', period='7d'):
    """
    Fetch historical S&P 500 data using yfinance.
    Args:
        interval (str): Data interval (e.g., '1d', '5m').
        period (str): Data period (e.g., '1y', '5y').
    Returns:
        pd.DataFrame: Historical data for S&P 500.
    """
    sp500 = yf.Ticker("^GSPC")
    hist = sp500.history(interval=interval, period=period)
    return hist

def get_sp500_data_for_lstm():
    """
    Get 5 years of daily S&P 500 data for LSTM model training.
    Returns:
        pd.DataFrame: Historical daily data for 5 years.
    """
    return get_sp500_data(interval='1d', period='5y')

def get_sp500_data_for_live():
    """
    Get recent S&P 500 data for live prediction (1-minute interval, 7 days).
    Returns:
        pd.DataFrame: Recent minute-level data.
    """
    return get_sp500_data(interval='1m', period='7d')

# --------------- SENTIMENT (LLM) ------------

def fetch_sentiment_finbert():
    """
    Fetch financial news headlines and compute average sentiment using FinBERT.
    Returns:
        float: Average sentiment score (-1=Negative, 0=Neutral, 1=Positive).
    """

    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    news_sources = [
        {
            "url": "https://www.marketwatch.com/latest-news",
            "tag": "h3"
        },
        {
            "url": "https://www.reuters.com/markets/",
            "tag": "h3"
        },
        {
            "url": "https://www.cnbc.com/world/?region=world",
            "tag": "a"  # CNBC headlines are often in <a> tags with class 'Card-title'
        },
        {
            "url": "https://www.bloomberg.com/markets",
            "tag": "h1"  # Bloomberg main headlines are in <h1>
        },
        {
            "url": "https://www.ft.com/markets",
            "tag": "a"  # FT headlines are often in <a> tags
        }
    ]

    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for source in news_sources:
        try:
            r = requests.get(source["url"], headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            # For some sources, you may want to filter by class for more accuracy
            tags = soup.find_all(source["tag"], limit=5)
            headlines += [tag.get_text(strip=True) for tag in tags]
        except Exception as e:
            print(f"Error fetching {source['url']}: {e}")

    headlines = [h for h in headlines if h.strip()]  # Remove empty headlines

    if not headlines:
        return 0  # or np.nan, or handle as you wish

    results = classifier(headlines)
    score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    scores = [score_map[r['label']] for r in results]
    return np.mean(scores)

# --------------- LSTM MODEL -----------------
def train_lstm_model(hist):
    """
    Train an LSTM model on historical closing price data.
    Args:
        hist (pd.DataFrame): Historical price data with 'Close' column.
    Returns:
        model (Sequential): Trained Keras LSTM model.
        scaler (MinMaxScaler): Scaler fitted to the data.
    """
    data = hist['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60,1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model, scaler

def predict_lstm(model, scaler, hist):
    """
    Predict the next closing price using the trained LSTM model.
    Args:
        model (Sequential): Trained LSTM model.
        scaler (MinMaxScaler): Fitted scaler.
        hist (pd.DataFrame): Recent price data with 'Close' column.
    Returns:
        float: Predicted next closing price.
    """
    last_60 = hist['Close'].values[-60:].reshape(-1, 1)
    last_scaled = scaler.transform(last_60)
    X_test = np.expand_dims(last_scaled, axis=(0, -1))
    pred_scaled = model.predict(X_test, verbose=0)
    prediction = scaler.inverse_transform(pred_scaled)[0][0]
    return prediction

# --------------- EMAIL ALERT ---------------


# ----------- UPDATE THREAD ------------------
hist_daily = get_sp500_data_for_lstm()
model_lstm, scaler_lstm = train_lstm_model(hist_daily)

def save_data_to_csv():
    """
    Save the current dashboard data to a CSV file.
    """
    with open('dashboard_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'actual', 'predicted', 'sentiment_score'])
        for t, a, p, s in zip(data_store['timestamps'], data_store['actual'], data_store['predicted'], data_store['sentiment_score']):
            writer.writerow([t, a, p, s])

def load_data_from_csv():
    """
    Load dashboard data from a CSV file into the data_store.
    """
    try:
        df = pd.read_csv('dashboard_data.csv', parse_dates=['timestamp'])
        data_store['timestamps'] = list(df['timestamp'])
        data_store['actual'] = list(df['actual'])
        data_store['predicted'] = list(df['predicted'])
        data_store['sentiment_score'] = list(df['sentiment_score'])
    except Exception:
        pass

# Load data at startup
load_data_from_csv()

def update_data():
    """
    Periodically fetch new data, update predictions, and save to CSV.
    """
    while True:
        hist = get_sp500_data_for_live()
        sentiment = fetch_sentiment_finbert()
        prediction_lstm = predict_lstm(model_lstm, scaler_lstm, hist)
        prediction = prediction_lstm + sentiment * 10

        actual = hist["Close"].iloc[-1]
        now = dt.datetime.now()

        data_store["timestamps"].append(now)
        data_store["actual"].append(actual)
        data_store["predicted"].append(prediction)
        data_store["sentiment_score"].append(sentiment)
        save_data_to_csv()

        # if abs(prediction - actual) > 10:
        #     send_alert(prediction, actual)

        time.sleep(60)  # 5 min

# ---------------- DASH APP ------------------
app = Dash(__name__)
app.title = "S&P 500 Prediction Dashboard"

app.layout = html.Div([
    html.H1("Live S&P 500 Prediction Dashboard"),
    dcc.Graph(id='trend-graph'),
    dcc.Interval(id='interval', interval=60*1000, n_intervals=0)
])

@app.callback(
    Output('trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_graph(n):
    """
    Dash callback to update the trend graph with actual and predicted values.
    Args:
        n (int): Number of intervals elapsed (not used).
    Returns:
        plotly.graph_objs.Figure: Updated graph figure.
    """
    if len(data_store["timestamps"]) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_store["timestamps"], y=data_store["actual"],
                             mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=data_store["timestamps"], y=data_store["predicted"],
                             mode='lines+markers', name='Predicted'))

    fig.update_layout(title='S&P 500 Actual vs Predicted Every 5 Minutes',
                      xaxis_title='Timestamp',
                      yaxis_title='Price',
                      template='plotly_dark')
    return fig

# ----------------- RUN ----------------------
if __name__ == "__main__":
    threading.Thread(target=update_data, daemon=True).start()
    app.run(debug=True, port=8050)
