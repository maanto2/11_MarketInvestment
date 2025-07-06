# marketPrediction4_optimized.py
"""
Optimized version of market prediction with performance improvements:
- Lazy loading of ML models
- Efficient data fetching with caching
- Reduced memory usage
- Better error handling and timeouts
- Optimized sentiment analysis
"""
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
import smtplib
import datetime as dt
import threading
import time
import csv
from keys import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, FRED_API_KEY
import pytz
from datetime import datetime, time as dtime, timezone
from fredapi import Fred
import functools
import logging
import concurrent.futures
from collections import deque
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add your Telegram bot token and chat ID here
TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN 
TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID
FRED_API_KEY = FRED_API_KEY

# ---------------- CONFIG -------------------
MAX_DATA_POINTS = 1000  # Limit data points to prevent memory bloat
CACHE_DURATION = 1800  # 30 minutes cache for expensive operations

# --------------- GLOBAL STORE --------------
class OptimizedDataStore:
    def __init__(self, max_points=MAX_DATA_POINTS):
        self.max_points = max_points
        self.timestamps = deque(maxlen=max_points)
        self.actual = deque(maxlen=max_points)
        self.predicted = deque(maxlen=max_points)
        self.sentiment_score = deque(maxlen=max_points)
        self._lock = threading.Lock()
    
    def add_data_point(self, timestamp, actual, predicted, sentiment):
        with self._lock:
            self.timestamps.append(timestamp)
            self.actual.append(actual)
            self.predicted.append(predicted)
            self.sentiment_score.append(sentiment)
    
    def get_latest_data(self, count=100):
        with self._lock:
            return {
                'timestamps': list(self.timestamps)[-count:],
                'actual': list(self.actual)[-count:],
                'predicted': list(self.predicted)[-count:],
                'sentiment_score': list(self.sentiment_score)[-count:]
            }

data_store = OptimizedDataStore()
nifty_data_store = OptimizedDataStore()

# Lazy loading for expensive ML models
class LazyModelLoader:
    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._classifier = None
        self._lock = threading.Lock()
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            with self._lock:
                if self._tokenizer is None:
                    logger.info("Loading BERT tokenizer...")
                    from transformers import BertTokenizer
                    self._tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        return self._tokenizer
    
    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading BERT model...")
                    from transformers import BertForSequenceClassification
                    self._model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        return self._model
    
    @property
    def classifier(self):
        if self._classifier is None:
            with self._lock:
                if self._classifier is None:
                    logger.info("Initializing sentiment classifier...")
                    from transformers import pipeline
                    self._classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        return self._classifier

# Initialize lazy model loader
model_loader = LazyModelLoader()

# FRED API setup with caching
fred = Fred(api_key=FRED_API_KEY)

# --------------- CACHING DECORATOR ------------
def cache_result(ttl_seconds=CACHE_DURATION):
    def decorator(func):
        cache = {}
        def wrapper(*args, **kwargs):
            now = time.time()
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            if cache_key in cache and (now - cache[cache_key]['time'] < ttl_seconds):
                return cache[cache_key]['value']
            
            value = func(*args, **kwargs)
            cache[cache_key] = {'value': value, 'time': now}
            return value
        return wrapper
    return decorator

# --------------- OPTIMIZED DATA FUNCTIONS ------------
@cache_result(ttl_seconds=300)  # 5 minutes cache for market data
def get_sp500_data(interval='5m', period='7d'):
    """Fetch historical S&P 500 data using yfinance with caching."""
    try:
        sp500 = yf.Ticker("^GSPC")
        hist = sp500.history(interval=interval, period=period)
        return hist
    except Exception as e:
        logger.error(f"Error fetching S&P 500 data: {e}")
        return pd.DataFrame()

@cache_result(ttl_seconds=300)
def get_nifty_data(interval='5m', period='7d'):
    """Fetch historical NIFTY 50 data using yfinance with caching."""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(interval=interval, period=period)
        return hist
    except Exception as e:
        logger.error(f"Error fetching NIFTY data: {e}")
        return pd.DataFrame()

def get_sp500_data_for_lstm():
    return get_sp500_data(interval='1m', period='7d')

def get_sp500_data_for_live():
    return get_sp500_data(interval='1m', period='7d')

def get_nifty_data_for_lstm():
    return get_nifty_data(interval='1m', period='7d')

def get_nifty_data_for_live():
    return get_nifty_data(interval='1m', period='7d')

# --------------- OPTIMIZED SENTIMENT ANALYSIS ------------
@cache_result(ttl_seconds=900)  # 15 minutes cache for sentiment
def fetch_sentiment_finbert():
    """Optimized sentiment analysis with reduced API calls and better error handling."""
    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Reduced number of news sources for better performance
    news_sources = [
        {"url": "https://www.marketwatch.com/latest-news", "tag": "h3"},
        {"url": "https://www.reuters.com/markets/", "tag": "h3"},
        {"url": "https://www.cnbc.com/world/?region=world", "tag": "a"},
    ]
    
    # Use concurrent requests for faster fetching
    def fetch_source(source):
        try:
            r = requests.get(source["url"], headers=headers, timeout=5)
            soup = BeautifulSoup(r.text, 'html.parser')
            tags = soup.find_all(source["tag"], limit=2)  # Reduced limit
            return [tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True)]
        except Exception as e:
            logger.warning(f"Error fetching {source['url']}: {e}")
            return []
    
    # Use ThreadPoolExecutor for concurrent fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_source, news_sources))
        headlines = [headline for result in results for headline in result]

    if not headlines:
        logger.info("No headlines found. Returning neutral sentiment (0).")
        return 0.0

    try:
        # Use lazy-loaded classifier
        results = model_loader.classifier(headlines[:10])  # Limit to 10 headlines
        score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        scores = [score_map[r['label']] for r in results]
        return np.mean(scores)
    except Exception as e:
        logger.error(f"Error during sentiment classification: {e}")
        return 0.0

# --------------- OPTIMIZED LSTM MODEL -----------------
def train_lstm_model(hist):
    """Optimized LSTM model training with reduced epochs and better memory management."""
    if hist.empty:
        logger.warning("No data available for LSTM training")
        return None, None
    
    try:
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
        model.fit(X, y, epochs=15, batch_size=32, verbose=0)  # Reduced epochs and batch size

        return model, scaler
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return None, None

def predict_lstm(model, scaler, hist):
    """Predict the next closing price using the trained LSTM model."""
    if model is None or scaler is None or hist.empty:
        return None
    
    try:
        last_60 = hist['Close'].values[-60:].reshape(-1, 1)
        last_scaled = scaler.transform(last_60)
        X_test = np.expand_dims(last_scaled, axis=(0, -1))
        pred_scaled = model.predict(X_test, verbose=0)
        prediction = scaler.inverse_transform(pred_scaled)[0][0]
        return prediction
    except Exception as e:
        logger.error(f"Error in LSTM prediction: {e}")
        return None

# --------------- OPTIMIZED DATA PERSISTENCE ------------
def save_data_to_csv():
    """Save data to CSV with error handling."""
    try:
        data = data_store.get_latest_data()
        df = pd.DataFrame(data)
        df.to_csv('dashboard_data.csv', index=False)
        logger.info("S&P 500 data saved to CSV")
    except Exception as e:
        logger.error(f"Error saving S&P 500 data: {e}")

def save_nifty_data_to_csv():
    """Save NIFTY data to CSV with error handling."""
    try:
        data = nifty_data_store.get_latest_data()
        df = pd.DataFrame(data)
        df.to_csv('dashboard_data_nifty.csv', index=False)
        logger.info("NIFTY data saved to CSV")
    except Exception as e:
        logger.error(f"Error saving NIFTY data: {e}")

def load_data_from_csv():
    """Load data from CSV with error handling."""
    try:
        df = pd.read_csv('dashboard_data.csv', parse_dates=['timestamp'])
        for _, row in df.iterrows():
            data_store.add_data_point(
                row['timestamp'], row['actual'], 
                row['predicted'], row['sentiment_score']
            )
        logger.info("S&P 500 data loaded from CSV")
    except Exception as e:
        logger.warning(f"Could not load S&P 500 CSV: {e}")

def load_nifty_data_from_csv():
    """Load NIFTY data from CSV with error handling."""
    try:
        df = pd.read_csv('dashboard_data_nifty.csv', parse_dates=['timestamp'])
        for _, row in df.iterrows():
            nifty_data_store.add_data_point(
                row['timestamp'], row['actual'], 
                row['predicted'], row['sentiment_score']
            )
        logger.info("NIFTY data loaded from CSV")
    except Exception as e:
        logger.warning(f"Could not load NIFTY CSV: {e}")

# --------------- OPTIMIZED UPDATE FUNCTIONS ------------
def update_data():
    """Optimized data update function with better error handling and reduced frequency."""
    logger.info("Starting S&P 500 data update thread")
    
    # Load initial data
    load_data_from_csv()
    
    # Train initial model
    hist_daily = get_sp500_data_for_lstm()
    model_lstm, scaler_lstm = train_lstm_model(hist_daily)
    
    while True:
        try:
            # Check if market is open
            if not is_market_open():
                time.sleep(300)  # Sleep longer when market is closed
                continue
            
            # Get current data
            hist = get_sp500_data_for_live()
            if hist.empty:
                logger.warning("No S&P 500 data available")
                time.sleep(60)
                continue
            
            # Get sentiment
            sentiment = fetch_sentiment_finbert()
            
            # Make prediction
            current_price = hist['Close'].iloc[-1]
            prediction = predict_lstm(model_lstm, scaler_lstm, hist)
            
            if prediction is not None:
                # Add data point
                data_store.add_data_point(
                    datetime.now(), current_price, prediction, sentiment
                )
                
                # Save to CSV periodically
                if len(data_store.timestamps) % 10 == 0:  # Save every 10 points
                    save_data_to_csv()
                
                logger.info(f"S&P 500: Actual={current_price:.2f}, Predicted={prediction:.2f}, Sentiment={sentiment:.2f}")
            
            # Retrain model periodically
            if len(data_store.timestamps) % 100 == 0:  # Retrain every 100 points
                logger.info("Retraining S&P 500 LSTM model...")
                model_lstm, scaler_lstm = train_lstm_model(hist_daily)
            
            time.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error(f"Error in S&P 500 update: {e}")
            time.sleep(60)

def update_nifty_data():
    """Optimized NIFTY data update function."""
    logger.info("Starting NIFTY 50 data update thread")
    
    # Load initial data
    load_nifty_data_from_csv()
    
    # Train initial model
    hist_daily = get_nifty_data_for_lstm()
    model_lstm, scaler_lstm = train_lstm_model(hist_daily)
    
    while True:
        try:
            # Check if Indian market is open
            if not is_india_market_open():
                time.sleep(300)  # Sleep longer when market is closed
                continue
            
            # Get current data
            hist = get_nifty_data_for_live()
            if hist.empty:
                logger.warning("No NIFTY data available")
                time.sleep(60)
                continue
            
            # Get sentiment (can reuse global sentiment or fetch India-specific)
            sentiment = fetch_sentiment_finbert()
            
            # Make prediction
            current_price = hist['Close'].iloc[-1]
            prediction = predict_lstm(model_lstm, scaler_lstm, hist)
            
            if prediction is not None:
                # Add data point
                nifty_data_store.add_data_point(
                    datetime.now(), current_price, prediction, sentiment
                )
                
                # Save to CSV periodically
                if len(nifty_data_store.timestamps) % 10 == 0:
                    save_nifty_data_to_csv()
                
                logger.info(f"NIFTY 50: Actual={current_price:.2f}, Predicted={prediction:.2f}, Sentiment={sentiment:.2f}")
            
            # Retrain model periodically
            if len(nifty_data_store.timestamps) % 100 == 0:
                logger.info("Retraining NIFTY LSTM model...")
                model_lstm, scaler_lstm = train_lstm_model(hist_daily)
            
            time.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error(f"Error in NIFTY update: {e}")
            time.sleep(60)

# --------------- MARKET STATUS FUNCTIONS ------------
def is_market_open():
    """Check if US market is open."""
    try:
        now = datetime.now(pytz.timezone('US/Eastern'))
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        return True  # Assume open if error

def is_india_market_open():
    """Check if Indian market is open."""
    try:
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except Exception as e:
        logger.error(f"Error checking Indian market status: {e}")
        return True  # Assume open if error

# --------------- TELEGRAM NOTIFICATIONS ------------
def send_telegram_message(message, bot_token, chat_id):
    """Send Telegram message with error handling."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": message}
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")

# Load initial data
if __name__ == "__main__":
    load_data_from_csv()
    load_nifty_data_from_csv() 