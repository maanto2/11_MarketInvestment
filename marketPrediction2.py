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
import pytz
from datetime import datetime, time as dtime
from keys import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Add your Telegram bot token and chat ID here
TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN 
TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID
# ---------------- CONFIG -------------------

# --------------- GLOBAL STORE --------------
data_store = {
    "actual": [],
    "predicted": [],
    "timestamps": [],
    "sentiment_score": [],
    "sentiment_summaries": []
}

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

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
    Get 6 months of daily S&P 500 data for LSTM model training.
    Returns:
        pd.DataFrame: Historical daily data for 6 months.
    """
    return get_sp500_data(interval='1m', period='7d')

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
    Fetch market sentiment from multiple news sources using FinBERT.
    Returns:
        tuple: (average_sentiment_score, summary_of_headlines)
    """
    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    news_sources = [
        {"url": "https://www.marketwatch.com/latest-news", "tag": "h3"},
        {"url": "https://www.reuters.com/markets/", "tag": "h3"},
        {"url": "https://www.cnbc.com/world/?region=world", "tag": "a"},
        {"url": "https://www.bloomberg.com/markets", "tag": "h1"},
        {"url": "https://www.ft.com/markets", "tag": "a"},]
    
    for source in news_sources:
        try:
            r = requests.get(source["url"], headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            tags = soup.find_all(source["tag"], limit=5)
            headlines += [tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True)]
        except Exception as e:
            print(f"Error fetching {source['url']}: {e}")
            continue  # skip to next source

    if not headlines:
        print("No headlines found. Returning neutral sentiment (0).")
        return 0.0, "No headlines found."

    try:
        results = classifier(headlines)
        score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        scores = [score_map[r['label']] for r in results]
        # Compose a short summary of the most influential headlines
        summary = []
        for h, r in zip(headlines, results):
            if r['label'] != 'Neutral':
                summary.append(f"{r['label']}: {h}")
        summary_text = '\n'.join(summary[:3]) if summary else 'No strong sentiment detected.'
        return np.mean(scores), summary_text
    except Exception as e:
        print(f"Error during sentiment classification: {e}")
        return 0.0, "Sentiment classification error."

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
    model.fit(X, y, epochs=30, batch_size=64, verbose=0)

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
    Append the latest data point to CSV.
    """
    file_exists = os.path.isfile('dashboard_data.csv')
    with open('dashboard_data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'actual', 'predicted', 'sentiment_score'])
        if data_store['timestamps']:
            writer.writerow([
                data_store['timestamps'][-1],
                data_store['actual'][-1],
                data_store['predicted'][-1],
                data_store['sentiment_score'][-1]
            ])

def load_data_from_csv():
    """
    Load dashboard data from a CSV file into the data_store.
    """
    try:
        df = pd.read_csv('dashboard_data.csv')  # assumes header row exists
        if df['timestamp'].iloc[0] == 'timestamp':
            df = df.iloc[1:]  # skip header row if duplicated as data
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        data_store['timestamps'] = list(df['timestamp'])
        data_store['actual'] = list(df['actual'])
        data_store['predicted'] = list(df['predicted'])
        data_store['sentiment_score'] = list(df['sentiment_score'])
    except Exception as e:
        print(f"[CSV Load Error] {e}")

# Load data at startup
load_data_from_csv()

def send_telegram_message(message, bot_token, chat_id):
    """
    Send a message to a Telegram chat using a bot.
    Args:
        message (str): The message to send.
        bot_token (str): Telegram bot token.
        chat_id (str): Telegram chat ID.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

def update_data():
    global model_lstm, scaler_lstm
    retrain_interval = 1440
    cycle_count = 0
    after_hours_sentiment = 0.0
    after_hours_summary = ""
    while True:
        start_time = time.time()
        try:
            if is_market_open():
                hist = get_sp500_data_for_live()
                sentiment, sentiment_summary = fetch_sentiment_finbert()
                # Add after-hours sentiment to today's sentiment if available
                sentiment += after_hours_sentiment
                prediction_lstm = predict_lstm(model_lstm, scaler_lstm, hist)
                sentiment_weight = (hist['Close'].iloc[-1] * 0.005)
                prediction = prediction_lstm + sentiment * sentiment_weight

                actual = hist["Close"].iloc[-1]
                now = dt.datetime.now()

                # Predict for the next minute and store it with a placeholder for actual
                next_timestamp = now + dt.timedelta(minutes=1)
                data_store["timestamps"].append(next_timestamp)
                data_store["predicted"].append(prediction)
                data_store["actual"].append(None)  # Placeholder for actual value
                data_store["sentiment_score"].append(sentiment)
                data_store["sentiment_summaries"].append(sentiment_summary)
                save_data_to_csv()

                # On the next loop, update the last None with the actual value
                if len(data_store["actual"]) > 1 and data_store["actual"][-2] is None:
                    data_store["actual"][-2] = actual
                    save_data_to_csv()

                if abs(sentiment) >= 0.6:
                    direction = "POSITIVE" if sentiment > 0 else "NEGATIVE"
                    message = f"Market sentiment is {direction} ({sentiment:.2f}) at {now.strftime('%Y-%m-%d %H:%M:%S')}! Consider short-term trading."
                    send_telegram_message(message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

                cycle_count += 1
                if cycle_count % retrain_interval == 0:
                    print("Retraining LSTM model with latest data...")
                    hist_daily = get_sp500_data_for_lstm()
                    model_lstm, scaler_lstm = train_lstm_model(hist_daily)
                    print("Retraining complete.")
            else:
                # After-hours: do in-depth S&P 500 component analysis
                after_hours_summary, after_hours_sentiment = analyze_sp500_components()
                print("[After Hours Research]", after_hours_summary)
                # Optionally, update dashboard with after-hours summary
                data_store["sentiment_summaries"].append("After Hours Research:\n" + after_hours_summary)
                save_data_to_csv()
                time.sleep(300)  # Sleep 5 minutes during after-hours
                continue
        except Exception as e:
            print(f"[Update Thread Error] {e}")
        elapsed = time.time() - start_time
        print(f"[Cycle Runtime] {elapsed:.2f} seconds")
        time.sleep(max(0, 60 - elapsed))

# ---------------- DASH APP ------------------
app = Dash(__name__)
app.title = "S&P 500 Prediction Dashboard"

app.layout = html.Div([
    html.H1("Live S&P 500 Prediction Dashboard"),
    dcc.Graph(id='trend-graph'),
    html.Div(id='sentiment-summary', style={'whiteSpace': 'pre-line', 'margin': '20px', 'fontSize': '16px', 'color': '#FFD700'}),
    dcc.Interval(id='interval', interval=60*1000, n_intervals=0)
])

@app.callback(
    [Output('trend-graph', 'figure'), Output('sentiment-summary', 'children')],
    [Input('interval', 'n_intervals')]
)
def update_graph(n):
    if len(data_store["timestamps"]) == 0:
        return go.Figure(), ""

    window = 5
    predicted_smoothed = pd.Series(data_store["predicted"]).rolling(window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_store["timestamps"], y=data_store["actual"],
                             mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=data_store["timestamps"], y=predicted_smoothed,
                             mode='lines+markers', name='Predicted (Smoothed)'))

    fig.update_layout(title='S&P 500 Actual vs Predicted (Smoothed) Every 5 Minutes',
                      xaxis_title='Timestamp',
                      yaxis_title='Price',
                      template='plotly_dark')

    # Show the latest sentiment summary
    sentiment_summary = data_store.get('sentiment_summaries', [])
    summary_text = sentiment_summary[-1] if sentiment_summary else "No sentiment summary available."
    return fig, summary_text

# S&P 500 market hours (Eastern Time)
def is_market_open():
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    is_weekday = now_et.weekday() < 5
    open_time = dtime(9, 30)
    close_time = dtime(16, 0)
    return is_weekday and open_time <= now_et.time() <= close_time

# Get S&P 500 tickers (static list for demo, can be fetched from Wikipedia or yfinance)
SP500_TICKERS = [
    # ... (add a subset for demo, or fetch dynamically for full list)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'NVDA', 'JPM', 'V', 'UNH', 'HD', 'PG', 'MA', 'DIS', 'BAC', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'KO', 'PFE', 'T', 'CSCO', 'PEP', 'ABT', 'CRM', 'XOM', 'CVX', 'WMT', 'INTC', 'MCD', 'MDT', 'COST', 'DHR', 'NKE', 'LLY', 'TMO', 'NEE', 'TXN', 'LIN', 'UNP', 'HON', 'PM', 'ORCL', 'AMGN', 'QCOM', 'IBM', 'ACN', 'AVGO', 'LOW', 'SBUX', 'MMM', 'GE', 'CAT', 'GS', 'BLK', 'AXP', 'SPGI', 'PLD', 'ISRG', 'MDLZ', 'CB', 'AMT', 'SYK', 'GILD', 'ZTS', 'CI', 'DE', 'DUK', 'MMC', 'ADI', 'C', 'SO', 'USB', 'PNC', 'TGT', 'BDX', 'CL', 'SHW', 'ICE', 'APD', 'ITW', 'MO', 'BKNG', 'FIS', 'ADP', 'EW', 'VRTX', 'REGN', 'AON', 'ETN', 'ECL', 'NSC', 'FDX', 'EMR', 'PSA', 'AIG', 'HUM', 'PSX', 'D', 'AEP', 'ALL', 'AFL', 'A', 'BAX', 'BMY', 'COF', 'DOV', 'EOG', 'EXC', 'F', 'GM', 'HCA', 'KMB', 'LHX', 'LMT', 'MET', 'MS', 'MTB', 'NOC', 'PGR', 'PRU', 'RF', 'RMD', 'SRE', 'STT', 'TFC', 'TRV', 'VLO', 'WBA', 'WELL', 'WMB', 'WST', 'ZBH'
]

def analyze_sp500_components():
    """
    Fetch and analyze all S&P 500 components for after-hours research.
    Returns a summary string for dashboard and a numeric score for next day sentiment.
    """
    try:
        data = yf.download(SP500_TICKERS, period='1d', group_by='ticker', threads=True)
        gainers = []
        losers = []
        sector_perf = {}
        for ticker in SP500_TICKERS:
            try:
                close = data[ticker]['Close'].iloc[-1]
                open_ = data[ticker]['Open'].iloc[-1]
                change = (close - open_) / open_ * 100
                if change > 2:
                    gainers.append(f"{ticker} (+{change:.2f}%)")
                elif change < -2:
                    losers.append(f"{ticker} ({change:.2f}%)")
            except Exception:
                continue
        summary = f"Top Gainers: {', '.join(gainers[:5])}\nTop Losers: {', '.join(losers[:5])}"
        # Simple sentiment: more gainers = positive, more losers = negative
        score = (len(gainers) - len(losers)) / len(SP500_TICKERS)
        return summary, score
    except Exception as e:
        return f"[SP500 Analysis Error] {e}", 0.0



