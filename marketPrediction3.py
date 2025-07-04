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
from keys import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,FRED_API_KEY
import pytz
from datetime import datetime, time as dtime, timezone
from fredapi import Fred
import functools
import logging
import concurrent.futures

# Add your Telegram bot token and chat ID here
TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN 
TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID
FRED_API_KEY = FRED_API_KEY
# ---------------- CONFIG -------------------

# --------------- GLOBAL STORE --------------
data_store = {
    "actual": [],
    "predicted": [],
    "timestamps": [],
    "sentiment_score": []
}

tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# FRED API setup
# You can get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html
# For public data, some endpoints work without a key, but you can add your key here for reliability.
fred = Fred(api_key=FRED_API_KEY)

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
    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    news_sources = [
        # US/global
        {"url": "https://www.marketwatch.com/latest-news", "tag": "h3"},
        {"url": "https://www.reuters.com/markets/", "tag": "h3"},
        {"url": "https://www.cnbc.com/world/?region=world", "tag": "a"},
        {"url": "https://www.bloomberg.com/markets", "tag": "h1"},
        {"url": "https://www.ft.com/markets", "tag": "a"},
        # China/Asia
        {"url": "https://www.scmp.com/business/china-business", "tag": "a"},
        {"url": "https://asia.nikkei.com/Business/Markets", "tag": "a"},
        {"url": "https://www.caixinglobal.com/markets/", "tag": "a"},
        {"url": "https://www.reuters.com/markets/asia/", "tag": "h3"},
    ]
    # Major Chinese companies (Yahoo Finance news pages)
    chinese_companies = ["BABA", "TCEHY", "JD", "BIDU", "NIO", "PDD", "LI", "XPEV"]
    for ticker in chinese_companies:
        news_sources.append({
            "url": f"https://finance.yahoo.com/quote/{ticker}/news", "tag": "h3"
        })

    for source in news_sources:
        try:
            r = requests.get(source["url"], headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            tags = soup.find_all(source["tag"], limit=2)  # limit to 2 headlines per source
            headlines += [tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True)]
        except Exception as e:
            print(f"Error fetching {source['url']}: {e}")
            continue  # skip to next source

    if not headlines:
        print("No headlines found. Returning neutral sentiment (0).")
        return 0.0

    try:
        results = classifier(headlines)
        score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        scores = [score_map[r['label']] for r in results]
        return np.mean(scores)
    except Exception as e:
        print(f"Error during sentiment classification: {e}")
        return 0.0

# Caching decorator for slow/rarely-changing sources
def cache_result(ttl_seconds=1800):
    def decorator(func):
        cache = {}
        def wrapper(*args, **kwargs):
            now = time.time()
            if 'value' in cache and (now - cache['time'] < ttl_seconds):
                return cache['value']
            value = func(*args, **kwargs)
            cache['value'] = value
            cache['time'] = now
            return value
        return wrapper
    return decorator

# Apply caching to slow/rarely-changing sources
@cache_result(ttl_seconds=1800)  # 30 minutes
def fetch_sector_performance_data():
    try:
        # Select Sector SPDR ETFs (XLF, XLK, XLE, etc.)
        sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLI', 'XLC', 'XLRE', 'XLU', 'XLB', 'XLP']
        sector_perf = {}
        for etf in sector_etfs:
            hist = yf.Ticker(etf).history(period='5d')
            perf = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
            sector_perf[etf] = perf
        return sector_perf
    except Exception as e:
        print(f"[Sector Performance Fetch Error] {e}")
        return None

@cache_result(ttl_seconds=1800)  # 30 minutes
def fetch_earnings_season_data():
    try:
        url = 'https://finance.yahoo.com/calendar/earnings/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        section = soup.find('section', {'data-test': 'earnings-summary'})
        if section:
            summary = section.get_text(strip=True)
            return {'summary': summary}
        # Fallback: try to get the first table or headline if section is missing
        table = soup.find('table')
        if table:
            rows = table.find_all('tr')
            if rows:
                first_row = ' | '.join([td.get_text(strip=True) for td in rows[0].find_all('td')])
                return {'summary': f"Earnings table (first row): {first_row}"}
        headline = soup.find('h1')
        if headline:
            return {'summary': headline.get_text(strip=True)}
        print("[Earnings Fetch Error] Earnings summary section and fallback not found.")
        return None
    except Exception as e:
        print(f"[Earnings Fetch Error] {e}")
        return None

@cache_result(ttl_seconds=1800)  # 30 minutes

def fetch_sentiment_china_news():
    """
    Fetch and analyze sentiment from China/Asia news and major Chinese companies (for caching).
    Returns a sentiment score.
    """
    headlines = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    news_sources = [
        {"url": "https://www.scmp.com/business/china-business", "tag": "a"},
        {"url": "https://asia.nikkei.com/Business/Markets", "tag": "a"},
        {"url": "https://www.caixinglobal.com/markets/", "tag": "a"},
        {"url": "https://www.reuters.com/markets/asia/", "tag": "h3"},
    ]
    # Major Chinese companies (Yahoo Finance news pages)
    chinese_companies = ["BABA", "TCEHY", "JD", "BIDU", "NIO", "PDD", "LI", "XPEV"]
    for ticker in chinese_companies:
        news_sources.append({
            "url": f"https://finance.yahoo.com/quote/{ticker}/news", "tag": "h3"
        })
    for source in news_sources:
        try:
            r = requests.get(source["url"], headers=headers, timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            tags = soup.find_all(source["tag"], limit=2)  # limit to 2 headlines per source
            headlines += [tag.get_text(strip=True) for tag in tags if tag.get_text(strip=True)]
        except Exception as e:
            print(f"Error fetching {source['url']}: {e}")
            continue
    if not headlines:
        print("No China/Asia headlines found. Returning neutral sentiment (0).")
        return 0.0
    try:
        results = classifier(headlines)
        score_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        scores = [score_map[r['label']] for r in results]
        return np.mean(scores)
    except Exception as e:
        print(f"Error during China sentiment classification: {e}")
        return 0.0

# In fetch_sentiment_finbert, call fetch_sentiment_china_news() and combine with global sentiment
old_fetch_sentiment_finbert = fetch_sentiment_finbert

def fetch_sentiment_finbert():
    global_sentiment = old_fetch_sentiment_finbert()
    china_sentiment = fetch_sentiment_china_news()
    # Weighted average: 70% global, 30% China/Asia
    return 0.7 * global_sentiment + 0.3 * china_sentiment

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
    Save the entire dashboard data to a CSV file, including all actual values, rounding decimals to 2 digits.
    """
    file_exists = os.path.isfile('dashboard_data.csv')
    with open('dashboard_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'actual', 'predicted', 'sentiment_score'])
        for t, a, p, s in zip(data_store['timestamps'], data_store['actual'], data_store['predicted'], data_store['sentiment_score']):
            # Round floats to 2 decimal places if not None, else write as is
            a_ = round(a, 2) if isinstance(a, float) and a is not None else a
            p_ = round(p, 2) if isinstance(p, float) and p is not None else p
            s_ = round(s, 2) if isinstance(s, float) and s is not None else s
            writer.writerow([t, a_, p_, s_])

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

def is_market_open():
    us_now = get_us_eastern_time_online()
    is_weekday = us_now.weekday() < 5
    open_time = dtime(9, 30)
    close_time = dtime(16, 0)
    return is_weekday and open_time <= us_now.time() <= close_time

def get_us_eastern_time_online():
    try:
        r = requests.get('http://worldtimeapi.org/api/timezone/America/New_York', timeout=5)
        if r.status_code == 200:
            data = r.json()
            dt_str = data['datetime']
            # Example: '2025-07-04T10:15:30.123456-04:00'
            from dateutil import parser
            return parser.isoparse(dt_str)
    except Exception as e:
        print(f"[Time API Error] {e}")
    # Fallback to local system time if API fails
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)

def get_local_and_us_time():
    """
    Returns current local system time and US/Eastern time as datetime objects.
    """
    local_now = datetime.now().astimezone()
    eastern = pytz.timezone('US/Eastern')
    us_now = datetime.now(eastern)
    return local_now, us_now

# Update is_market_open to also return local and US/Eastern time for dashboard

def is_market_open_with_times():
    eastern = pytz.timezone('US/Eastern')
    us_now = datetime.now(eastern)
    is_weekday = us_now.weekday() < 5
    open_time = dtime(9, 30)
    close_time = dtime(16, 0)
    market_open = is_weekday and open_time <= us_now.time() <= close_time
    local_now = datetime.now().astimezone()
    return market_open, local_now, us_now

def update_data():
    global model_lstm, scaler_lstm
    retrain_interval = 1440
    cycle_count = 0
    after_hours_sentiment = 0.0
    after_hours_summary = ""
    while True:
        try:
            cycle_start = time.perf_counter()
            if is_market_open():
                hist = get_sp500_data_for_live()
                sentiment = fetch_sentiment_finbert()
                sentiment += after_hours_sentiment
                prediction_lstm = predict_lstm(model_lstm, scaler_lstm, hist)
                sentiment_weight = (hist['Close'].iloc[-1] * 0.005)
                prediction = prediction_lstm + sentiment * sentiment_weight

                # --- Macro factors integration ---
                macro_factors = aggregate_macro_factors()
                macro_score = sum(1 for v in macro_factors.values() if v not in [None, 0, '', {}, []]) / len(macro_factors)
                if 'macro_score' not in data_store:
                    data_store['macro_score'] = []

                actual = hist["Close"].iloc[-1]
                now = dt.datetime.now()
                next_timestamp = now + dt.timedelta(minutes=1)
                data_store["timestamps"].append(next_timestamp)
                data_store["predicted"].append(prediction)
                data_store["actual"].append(None)
                data_store["sentiment_score"].append(sentiment)
                data_store["macro_score"].append(macro_score)
                save_data_to_csv()

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
                run_after_hours_analysis()
                time.sleep(300)
                continue
        except Exception as e:
            print(f"[Update Thread Error] {e}")
        cycle_elapsed = time.perf_counter() - cycle_start
        logging.info(f"update_data cycle took {cycle_elapsed:.3f} seconds")
        time.sleep(60)

def run_after_hours_analysis():
    """
    Run after-hours S&P 500 analysis in a background thread.
    Only run detailed research on weekends.
    Updates the data_store with research results when done.
    """
    def analysis_task():
        after_hours_summary, after_hours_sentiment = analyze_sp500_components()
        # Only run detailed research on weekends
        if datetime.now(pytz.timezone('US/Eastern')).weekday() >= 5:
            detailed_summary, detailed_score = analyze_sp500_detailed_parallel()
            after_hours_summary += "\n" + detailed_summary
            after_hours_sentiment += detailed_score
        print("[After Hours Research]", after_hours_summary)
        # Always append to sentiment_summaries
        if "sentiment_summaries" not in data_store:
            data_store["sentiment_summaries"] = []
        data_store["sentiment_summaries"].append("After Hours Research:\n" + after_hours_summary)
        save_data_to_csv()
    t = threading.Thread(target=analysis_task, daemon=True)
    t.start()

# ---------------- DASH APP ------------------
app = Dash(__name__)
app.title = "S&P 500 Prediction Dashboard"

app.layout = html.Div([
    html.H1("Live S&P 500 Prediction Dashboard"),
    html.Div(id='market-status-banner', style={'backgroundColor': '#222', 'color': 'yellow', 'padding': '8px', 'fontWeight': 'bold', 'fontSize': '18px', 'overflowX': 'auto', 'whiteSpace': 'nowrap'}),
    dcc.Graph(id='trend-graph'),
    html.Div(id='news-summary', style={'backgroundColor': '#111', 'color': 'white', 'padding': '12px', 'marginTop': '10px', 'maxHeight': '200px', 'overflowY': 'auto', 'fontSize': '15px', 'border': '1px solid #333'}),
    dcc.Interval(id='interval', interval=60*1000, n_intervals=0)
])

# Update the dashboard banner to show both times and market status
@app.callback(
    Output('market-status-banner', 'children'),
    [Input('interval', 'n_intervals')]
)
def update_market_status(n):
    market_open, local_now, us_now = is_market_open_with_times()
    local_str = local_now.strftime('%Y-%m-%d %H:%M:%S %Z')
    us_str = us_now.strftime('%Y-%m-%d %H:%M:%S %Z')
    if market_open:
        return f'🟢 Market is OPEN | Local time: {local_str} | US/Eastern: {us_str}'
    else:
        return f'🔴 Market is CLOSED | Local time: {local_str} | US/Eastern: {us_str}'

@app.callback(
    Output('trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_graph(n):
    if len(data_store["timestamps"]) == 0:
        return go.Figure()

    window = 5
    predicted_smoothed = pd.Series(data_store["predicted"]).rolling(window, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_store["timestamps"], y=data_store["actual"],
                             mode='lines+markers', name='Actual Price'))
    fig.add_trace(go.Scatter(x=data_store["timestamps"], y=predicted_smoothed,
                             mode='lines+markers', name='Predicted (Smoothed)'))
    # Remove sentiment score from the graph
    # Add macro score if present
    if "macro_score" in data_store and len(data_store["macro_score"]) == len(data_store["timestamps"]):
        fig.add_trace(go.Scatter(x=data_store["timestamps"], y=data_store["macro_score"],
                                 mode='lines+markers', name='Macro Score', yaxis='y2'))

    fig.update_layout(
        title='S&P 500 Actual vs Predicted with Macro Factors',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        yaxis2=dict(title='Macro Score', overlaying='y', side='right', showgrid=False),
        template='plotly_dark'
    )
    return fig

# S&P 500 tickers (short demo list, expand as needed)
# Reduce the number of tickers for after-hours analysis to the top 30 by market cap (example subset)
SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'UNH',
    'HD', 'PG', 'MA', 'DIS', 'BAC', 'VZ', 'ADBE', 'CMCSA', 'NFLX', 'KO',
    'PFE', 'T', 'CSCO', 'PEP', 'ABT', 'CRM', 'XOM', 'CVX', 'WMT', 'INTC'
]

def analyze_sp500_components():
    """
    Fetch and analyze all S&P 500 components for after-hours research.
    Returns a summary string for dashboard and a numeric score for next day sentiment.
    """
    try:
        # Reduce frequency: cache data for 10 minutes
        if not hasattr(analyze_sp500_components, '_last_fetch') or \
           (time.time() - getattr(analyze_sp500_components, '_last_fetch', 0)) > 600:
            analyze_sp500_components._cached_data = yf.download(
                SP500_TICKERS, period='1d', group_by='ticker', threads=True, progress=False, auto_adjust=False
            )
            analyze_sp500_components._last_fetch = time.time()
        data = analyze_sp500_components._cached_data
        gainers = []
        losers = []
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
        score = (len(gainers) - len(losers)) / len(SP500_TICKERS)
        return summary, score
    except Exception as e:
        return f"[SP500 Analysis Error] {e}", 0.0


def analyze_sp500_detailed_parallel():
    """
    Perform detailed analysis (quarterly profit, moving average, volume) for S&P 500 components in parallel.
    Returns a summary string and a numeric score for next day sentiment.
    """
    def analyze_ticker(ticker):
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period='6mo', interval='1d')
            info = data.info
            ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            avg_vol = hist['Volume'].rolling(window=20).mean().iloc[-1]
            profit = info.get('profitMargins', None)
            return {
                'ticker': ticker,
                'ma20': ma20,
                'avg_vol': avg_vol,
                'profit': profit
            }
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for res in executor.map(analyze_ticker, SP500_TICKERS):
            results.append(res)

    ma20_positive = [r for r in results if isinstance(r.get('ma20'), float) and r['ma20'] > 0]
    avg_vol_high = [r for r in results if isinstance(r.get('avg_vol'), float) and r['avg_vol'] > 1e6]
    profit_positive = [r for r in results if r.get('profit') and r['profit'] > 0]
    summary = (
        f"Tickers with positive 20-day MA: {len(ma20_positive)}\n"
        f"Tickers with avg volume > 1M: {len(avg_vol_high)}\n"
        f"Tickers with positive profit margin: {len(profit_positive)}\n"
    )
    score = (len(ma20_positive) + len(avg_vol_high) + len(profit_positive)) / (3 * len(SP500_TICKERS))
    return summary, score

# ----------------- MACROECONOMIC & MARKET FACTORS FETCHERS -----------------
def fetch_gdp_data():
    try:
        gdp = fred.get_series_latest_release('GDP')
        if gdp is not None and hasattr(gdp, 'iloc') and not gdp.empty:
            return float(gdp.iloc[-1])
        elif isinstance(gdp, (float, int)):
            return float(gdp)
        else:
            print("[GDP Fetch Error] No valid GDP data returned.")
            return None
    except Exception as e:
        print(f"[GDP Fetch Error] {e}")
        return None

def fetch_inflation_data():
    try:
        cpi = fred.get_series_latest_release('CPIAUCSL')
        if cpi is not None and hasattr(cpi, 'iloc') and not cpi.empty:
            return float(cpi.iloc[-1])
        elif isinstance(cpi, (float, int)):
            return float(cpi)
        else:
            print("[Inflation Fetch Error] No valid CPI data returned.")
            return None
    except Exception as e:
        print(f"[Inflation Fetch Error] {e}")
        return None
# Fed Policy (scrape Federal Reserve news headlines)
def fetch_fed_policy_data():
    try:
        url = 'https://www.federalreserve.gov/newsevents/pressreleases.htm'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.get_text(strip=True) for h in soup.select('.item .title')][:5]
        return {'headlines': headlines}
    except Exception as e:
        print(f"[Fed Policy Fetch Error] {e}")
        return None

# Global Events (reuse FinBERT/news logic for sentiment)
def fetch_global_events_data():
    try:
        sentiment = fetch_sentiment_finbert()
        return {'sentiment': sentiment}
    except Exception as e:
        print(f"[Global Events Fetch Error] {e}")
        return None

# Government Policy (scrape recent US government press releases)
def fetch_government_policy_data():
    try:
        url = 'https://www.whitehouse.gov/briefing-room/statements-releases/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        releases = [a.get_text(strip=True) for a in soup.select('article h2 a')][:5]
        return {'releases': releases}
    except Exception as e:
        print(f"[Gov Policy Fetch Error] {e}")
        return None

# Earnings Season (Yahoo Finance earnings calendar summary)
def fetch_earnings_season_data():
    try:
        url = 'https://finance.yahoo.com/calendar/earnings/'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        section = soup.find('section', {'data-test': 'earnings-summary'})
        if section:
            summary = section.get_text(strip=True)
            return {'summary': summary}
        # Fallback: try to get the first table or headline if section is missing
        table = soup.find('table')
        if table:
            rows = table.find_all('tr')
            if rows:
                first_row = ' | '.join([td.get_text(strip=True) for td in rows[0].find_all('td')])
                return {'summary': f"Earnings table (first row): {first_row}"}
        headline = soup.find('h1')
        if headline:
            return {'summary': headline.get_text(strip=True)}
        print("[Earnings Fetch Error] Earnings summary section and fallback not found.")
        return None
    except Exception as e:
        print(f"[Earnings Fetch Error] {e}")
        return None

def fetch_sector_performance_data():
    try:
        # Select Sector SPDR ETFs (XLF, XLK, XLE, etc.)
        sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLY', 'XLI', 'XLC', 'XLRE', 'XLU', 'XLB', 'XLP']
        sector_perf = {}
        for etf in sector_etfs:
            hist = yf.Ticker(etf).history(period='5d')
            perf = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
            sector_perf[etf] = perf
        return sector_perf
    except Exception as e:
        print(f"[Sector Performance Fetch Error] {e}")
        return None

def fetch_bond_yields_data():
    """
    Fetch latest bond yields (e.g., US 10Y Treasury).
    Returns:
        float or dict: Bond yield value(s).
    """
    try:
        # Example: 10-Year Treasury Constant Maturity Rate
        yield_10y = fred.get_series_latest_release('DGS10')
        return float(yield_10y) if yield_10y is not None else None
    except Exception as e:
        print(f"[Bond Yield Fetch Error] {e}")
        return None
def fetch_commodities_data():
    try:
        # Gold: 'GC=F', Oil: 'CL=F'
        gold = yf.Ticker('GC=F').history(period='1d')['Close'].iloc[-1]
        oil = yf.Ticker('CL=F').history(period='1d')['Close'].iloc[-1]
        return {'gold': float(gold), 'oil': float(oil)}
    except Exception as e:
        print(f"[Commodities Fetch Error] {e}")
        return None

def fetch_currency_data():
    try:
        # US Dollar Index: 'DX-Y.NYB' or 'DXY'
        dxy = yf.Ticker('DX-Y.NYB').history(period='1d')['Close'].iloc[-1]
        return {'usd_index': float(dxy)}
    except Exception as e:
        print(f"[Currency Fetch Error] {e}")
        return None

def fetch_volatility_and_global_features():
    """
    Fetch S&P 500 futures, VIX, and major global indices for open prediction features.
    Returns:
        dict: {'spx_futures': float, 'vix': float, 'nikkei': float, 'dax': float, ...}
    """
    try:
        features = {}
        # S&P 500 E-mini Futures (CME): 'ES=F'
        try:
            features['spx_futures'] = float(yf.Ticker('ES=F').history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            print(f"[Futures Fetch Error] {e}")
            features['spx_futures'] = None
        # VIX Volatility Index: '^VIX'
        try:
            features['vix'] = float(yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            print(f"[VIX Fetch Error] {e}")
            features['vix'] = None
        # Nikkei 225: '^N225'
        try:
            features['nikkei'] = float(yf.Ticker('^N225').history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            print(f"[Nikkei Fetch Error] {e}")
            features['nikkei'] = None
        # DAX (Germany): '^GDAXI'
        try:
            features['dax'] = float(yf.Ticker('^GDAXI').history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            print(f"[DAX Fetch Error] {e}")
            features['dax'] = None
        # FTSE 100 (UK): '^FTSE'
        try:
            features['ftse'] = float(yf.Ticker('^FTSE').history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            print(f"[FTSE Fetch Error] {e}")
            features['ftse'] = None
        # Hang Seng (Hong Kong): '^HSI'
        try:
            features['hangseng'] = float(yf.Ticker('^HSI').history(period='1d')['Close'].iloc[-1])
        except Exception as e:
            print(f"[Hang Seng Fetch Error] {e}")
            features['hangseng'] = None
        return features
    except Exception as e:
        print(f"[Volatility/Global Fetch Error] {e}")
        return {}

# Add to macro aggregation
def aggregate_macro_factors():
    """
    Aggregate all macroeconomic and market factors into a single dictionary for analysis, in parallel, with per-source timeouts.
    Returns:
        dict: Aggregated macro/market data.
    """
    fetchers = {
        'gdp': fetch_gdp_data,
        'inflation': fetch_inflation_data,
        'fed_policy': fetch_fed_policy_data,
        'global_events': fetch_global_events_data,
        'sector_performance': fetch_sector_performance_data,
        'bond_yields': fetch_bond_yields_data,
        'commodities': fetch_commodities_data,
        'currency': fetch_currency_data,
        'government_policy': fetch_government_policy_data,
        'earnings_season': fetch_earnings_season_data,
        'volatility_global': fetch_volatility_and_global_features,
    }
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fetchers)) as executor:
        future_to_key = {executor.submit(run_with_timeout, fn, 10, None): key for key, fn in fetchers.items()}
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results[key] = None
                print(f"[aggregate_macro_factors] Error fetching {key}: {e}")
    return results

def run_with_timeout(func, timeout=10, default=None):
    """
    Run a function with a timeout. If it takes too long, return default.
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            print(f"[Timeout] {func.__name__} exceeded {timeout}s: {e}")
            return default

# Setup logging for profiling
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

def profile_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper

# ----------- CACHE SLOW FETCHERS -----------
# Place these at the very end of the file, after all function definitions
fetch_government_policy_data = cache_result(1800)(fetch_government_policy_data)
fetch_earnings_season_data = cache_result(1800)(fetch_earnings_season_data)
fetch_sector_performance_data = cache_result(1800)(fetch_sector_performance_data)
fetch_commodities_data = cache_result(1800)(fetch_commodities_data)
fetch_currency_data = cache_result(1800)(fetch_currency_data)

# ----------------- RUN ----------------------
if __name__ == "__main__":
    threading.Thread(target=update_data, daemon=True).start()
    run_after_hours_analysis()  # Start the after-hours analysis thread
    app.run(debug=True, port=8050)