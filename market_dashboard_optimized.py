# market_dashboard_optimized.py
"""
Optimized version of the market dashboard with performance improvements:
- Efficient data caching and management
- Optimized plotting with data limits
- Reduced CSV I/O operations
- Better memory management
"""
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import threading
import time
import os
from datetime import datetime, timedelta
import pytz
import requests
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimized data stores with size limits
MAX_DATA_POINTS = 1000  # Limit data points to prevent memory bloat

class OptimizedDataStore:
    def __init__(self, max_points=MAX_DATA_POINTS):
        self.max_points = max_points
        self.timestamps = deque(maxlen=max_points)
        self.actual = deque(maxlen=max_points)
        self.predicted = deque(maxlen=max_points)
        self.sentiment_score = deque(maxlen=max_points)
        self.last_update = None
        self._lock = threading.Lock()
    
    def update_data(self, timestamps, actual, predicted, sentiment_score):
        with self._lock:
            # Only update if we have new data
            if (len(timestamps) > 0 and 
                (self.last_update is None or 
                 (isinstance(timestamps[-1], (str, pd.Timestamp)) and 
                  pd.to_datetime(timestamps[-1]) > pd.to_datetime(self.last_update)))):
                
                # Convert to lists and add to deques
                self.timestamps.extend(timestamps)
                self.actual.extend(actual)
                self.predicted.extend(predicted)
                self.sentiment_score.extend(sentiment_score)
                
                # Update last_update with the latest timestamp
                if timestamps:
                    self.last_update = pd.to_datetime(timestamps[-1])
                logger.info(f"Updated data store with {len(timestamps)} new points")
    
    def get_recent_data(self, days=7):
        """Get recent data for plotting (last N days)"""
        with self._lock:
            if len(self.timestamps) == 0:
                return [], [], [], []
            
            # Convert to pandas for easier filtering
            df = pd.DataFrame({
                'timestamp': list(self.timestamps),
                'actual': list(self.actual),
                'predicted': list(self.predicted),
                'sentiment_score': list(self.sentiment_score)
            })
            
            # Ensure timestamp column is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to last N days
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            
            return (list(df['timestamp']), list(df['actual']), 
                   list(df['predicted']), list(df['sentiment_score']))

# Initialize optimized data stores
sp500_store = OptimizedDataStore()
nifty_store = OptimizedDataStore()

RESEARCHER_API_URL = "http://192.168.1.112:5000/api"

def fetch_api_data_with_cache(endpoint, cache_duration=30):
    """Fetch API data with simple caching to reduce redundant calls"""
    cache_key = f"{endpoint}_cache"
    current_time = time.time()
    
    # Check if we have cached data
    if hasattr(fetch_api_data_with_cache, cache_key):
        cached_data, cache_time = getattr(fetch_api_data_with_cache, cache_key)
        if current_time - cache_time < cache_duration:
            return cached_data
    
    try:
        r = requests.get(f"{RESEARCHER_API_URL}/{endpoint}", timeout=5)
        if r.status_code == 200:
            # Check if response is valid JSON
            try:
                data = r.json()
                
                # Validate data structure and clean NaN values
                if isinstance(data, dict) and 'timestamps' in data:
                    # Convert timestamps to pandas datetime if needed
                    if data['timestamps']:
                        data['timestamps'] = pd.to_datetime(data['timestamps'])
                    
                    # Clean NaN values from numeric fields
                    for field in ['actual', 'predicted', 'sentiment_score']:
                        if field in data and data[field]:
                            # Replace NaN with None or 0
                            cleaned_values = []
                            for val in data[field]:
                                if pd.isna(val) or val is None:
                                    cleaned_values.append(0.0)
                                else:
                                    cleaned_values.append(float(val))
                            data[field] = cleaned_values
                    
                    # Cache the result
                    setattr(fetch_api_data_with_cache, cache_key, (data, current_time))
                    return data
                else:
                    logger.warning(f"Invalid data structure from API: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                    return None
            except ValueError as json_error:
                logger.error(f"Invalid JSON response from API: {r.text[:100]}...")
                return None
    except requests.exceptions.ConnectionError:
        logger.warning(f"Cannot connect to researcher API at {RESEARCHER_API_URL}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {endpoint} from researcher API: {e}")
    return None

def efficient_csv_loader():
    """Load data from CSV files efficiently with error handling"""
    try:
        # Load S&P 500 data
        if os.path.exists('dashboard_data.csv'):
            df_sp500 = pd.read_csv('dashboard_data.csv', parse_dates=['timestamp'])
            if not df_sp500.empty:
                sp500_store.update_data(
                    df_sp500['timestamp'].tolist(),
                    df_sp500['actual'].tolist(),
                    df_sp500['predicted'].tolist(),
                    df_sp500['sentiment_score'].tolist()
                )
                logger.info(f"Loaded {len(df_sp500)} S&P 500 data points from CSV")
        else:
            logger.warning("dashboard_data.csv not found")
    except Exception as e:
        logger.warning(f"Could not load S&P 500 CSV: {e}")
    
    try:
        # Load NIFTY data
        if os.path.exists('dashboard_data_nifty.csv'):
            df_nifty = pd.read_csv('dashboard_data_nifty.csv', parse_dates=['timestamp'])
            if not df_nifty.empty:
                nifty_store.update_data(
                    df_nifty['timestamp'].tolist(),
                    df_nifty['actual'].tolist(),
                    df_nifty['predicted'].tolist(),
                    df_nifty['sentiment_score'].tolist()
                )
                logger.info(f"Loaded {len(df_nifty)} NIFTY data points from CSV")
        else:
            logger.warning("dashboard_data_nifty.csv not found")
    except Exception as e:
        logger.warning(f"Could not load NIFTY CSV: {e}")

def periodic_data_update():
    """Optimized periodic data update with reduced frequency"""
    while True:
        try:
            # Try API first, fallback to CSV
            sp500_api_data = fetch_api_data_with_cache('sp500')
            if sp500_api_data and sp500_api_data.get('timestamps'):
                try:
                    sp500_store.update_data(
                        sp500_api_data['timestamps'],
                        sp500_api_data['actual'],
                        sp500_api_data['predicted'],
                        sp500_api_data['sentiment_score']
                    )
                except Exception as e:
                    logger.error(f"Error updating S&P 500 data: {e}")
            else:
                logger.info("No S&P 500 API data available, using CSV fallback")
            
            nifty_api_data = fetch_api_data_with_cache('nifty')
            if nifty_api_data and nifty_api_data.get('timestamps'):
                try:
                    nifty_store.update_data(
                        nifty_api_data['timestamps'],
                        nifty_api_data['actual'],
                        nifty_api_data['predicted'],
                        nifty_api_data['sentiment_score']
                    )
                except Exception as e:
                    logger.error(f"Error updating NIFTY data: {e}")
            else:
                logger.info("No NIFTY API data available, using CSV fallback")
            
            # Always load from CSV as fallback
            try:
                efficient_csv_loader()
            except Exception as e:
                logger.error(f"Error in CSV loader: {e}")
                
        except Exception as e:
            logger.error(f"Error in periodic update: {e}")
            try:
                efficient_csv_loader()
            except Exception as csv_error:
                logger.error(f"Error in CSV fallback: {csv_error}")
        
        time.sleep(60)  # Reduced frequency from 30s to 60s

# Start background thread for data updates
threading.Thread(target=periodic_data_update, daemon=True).start()

# Initialize app
app = Dash(__name__)
app.title = "S&P 500 & NIFTY 50 Prediction Dashboard (Optimized)"

app.layout = html.Div([
    html.H1("Live S&P 500 & NIFTY 50 Prediction Dashboard (Optimized)"),
    html.Div([
        html.Div([
            html.H2("S&P 500"),
            dcc.Graph(id='trend-graph'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            html.H2("NIFTY 50 (India)"),
            dcc.Graph(id='nifty-trend-graph'),
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    dcc.Interval(id='interval', interval=120*1000, n_intervals=0)  # Reduced update frequency
])

def create_optimized_figure(timestamps, actual, predicted, title):
    """Create optimized plotly figure with reduced data points for better performance"""
    if len(timestamps) == 0:
        return go.Figure()
    
    # Limit data points for better performance
    max_points = 200
    if len(timestamps) > max_points:
        step = len(timestamps) // max_points
        timestamps = timestamps[::step]
        actual = actual[::step]
        predicted = predicted[::step]
    
    # Smooth predictions
    window = 5
    predicted_smoothed = pd.Series(predicted).rolling(window, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=actual,
        mode='lines+markers', 
        name='Actual Price',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, 
        y=predicted_smoothed,
        mode='lines+markers', 
        name='Predicted (Smoothed)',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Timestamp',
        yaxis_title='Price',
        template='plotly_dark',
        height=400,  # Fixed height for better performance
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output('trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_graph(n):
    timestamps, actual, predicted, sentiment = sp500_store.get_recent_data(days=7)
    return create_optimized_figure(timestamps, actual, predicted, 'S&P 500 Actual vs Predicted')

@app.callback(
    Output('nifty-trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_nifty_graph(n):
    timestamps, actual, predicted, sentiment = nifty_store.get_recent_data(days=7)
    return create_optimized_figure(timestamps, actual, predicted, 'NIFTY 50 Actual vs Predicted')

if __name__ == "__main__":
    # Load initial data
    efficient_csv_loader()
    
    # Run the app
    app.run(debug=False, port=8050)  # Disabled debug mode for better performance 