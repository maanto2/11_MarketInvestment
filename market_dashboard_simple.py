# market_dashboard_simple.py
"""
Simplified market dashboard that only uses CSV files - no API connections.
This eliminates all JSON parsing errors and API connection issues.
"""
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import threading
import time
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple data stores
class SimpleDataStore:
    def __init__(self):
        self.timestamps = []
        self.actual = []
        self.predicted = []
        self.sentiment_score = []
        self._lock = threading.Lock()
    
    def load_from_csv(self, filename):
        """Load data from CSV file"""
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename, parse_dates=['timestamp'])
                with self._lock:
                    self.timestamps = df['timestamp'].tolist()
                    self.actual = df['actual'].tolist()
                    self.predicted = df['predicted'].tolist()
                    self.sentiment_score = df['sentiment_score'].tolist()
                logger.info(f"Loaded {len(df)} data points from {filename}")
                return True
            else:
                logger.warning(f"File {filename} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return False
    
    def get_recent_data(self, days=7):
        """Get recent data for plotting"""
        with self._lock:
            if not self.timestamps:
                return [], [], [], []
            
            # Convert to DataFrame for filtering
            df = pd.DataFrame({
                'timestamp': self.timestamps,
                'actual': self.actual,
                'predicted': self.predicted,
                'sentiment_score': self.sentiment_score
            })
            
            # Filter to last N days
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[pd.to_datetime(df['timestamp']) >= cutoff_date]
            
            return (df['timestamp'].tolist(), df['actual'].tolist(), 
                   df['predicted'].tolist(), df['sentiment_score'].tolist())

# Initialize data stores
sp500_store = SimpleDataStore()
nifty_store = SimpleDataStore()

def load_all_data():
    """Load data from both CSV files"""
    sp500_loaded = sp500_store.load_from_csv('dashboard_data.csv')
    nifty_loaded = nifty_store.load_from_csv('dashboard_data_nifty.csv')
    
    if not sp500_loaded and not nifty_loaded:
        logger.error("No data files found!")
        return False
    
    return True

def periodic_reload():
    """Reload data from CSV files periodically"""
    while True:
        try:
            load_all_data()
        except Exception as e:
            logger.error(f"Error in periodic reload: {e}")
        time.sleep(60)  # Reload every minute

# Start background thread for data reloading
threading.Thread(target=periodic_reload, daemon=True).start()

# Initialize app
app = Dash(__name__)
app.title = "S&P 500 & NIFTY 50 Prediction Dashboard (Simple)"

app.layout = html.Div([
    html.H1("Live S&P 500 & NIFTY 50 Prediction Dashboard (Simple)"),
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
    dcc.Interval(id='interval', interval=120*1000, n_intervals=0)  # Update every 2 minutes
])

def create_simple_figure(timestamps, actual, predicted, title):
    """Create simple plotly figure"""
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
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output('trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_graph(n):
    timestamps, actual, predicted, sentiment = sp500_store.get_recent_data(days=7)
    return create_simple_figure(timestamps, actual, predicted, 'S&P 500 Actual vs Predicted')

@app.callback(
    Output('nifty-trend-graph', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_nifty_graph(n):
    timestamps, actual, predicted, sentiment = nifty_store.get_recent_data(days=7)
    return create_simple_figure(timestamps, actual, predicted, 'NIFTY 50 Actual vs Predicted')

if __name__ == "__main__":
    # Load initial data
    if load_all_data():
        logger.info("Dashboard started successfully with data loaded")
    else:
        logger.warning("Dashboard started but no data files found")
    
    # Run the app
    app.run(debug=False, port=8050) 