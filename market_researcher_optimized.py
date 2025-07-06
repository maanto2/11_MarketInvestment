# market_researcher_optimized.py
"""
Optimized version of the market researcher with performance improvements:
- Reduced API calls with caching
- Optimized data processing
- Better error handling and timeouts
- Memory-efficient data storage
"""
import os
import time
import threading
from marketPrediction4_optimized import (
    update_data, update_nifty_data, data_store, nifty_data_store
)
from flask import Flask, jsonify
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Cache for API responses
api_cache = {}
CACHE_DURATION = 300  # 5 minutes

def cache_response(func):
    """Decorator to cache API responses"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        current_time = time.time()
        
        # Check cache
        if cache_key in api_cache:
            cached_data, cache_time = api_cache[cache_key]
            if current_time - cache_time < CACHE_DURATION:
                logger.info(f"Returning cached data for {func.__name__}")
                return cached_data
        
        # Get fresh data
        result = func(*args, **kwargs)
        api_cache[cache_key] = (result, current_time)
        return result
    return wrapper

@cache_response
def get_sp500_data():
    """Get S&P 500 data with caching"""
    try:
        # Return the latest S&P 500 data (last 100 points)
        latest_data = data_store.get_latest_data(100)
        return {
            'timestamps': latest_data['timestamps'],
            'actual': latest_data['actual'],
            'predicted': latest_data['predicted'],
            'sentiment_score': latest_data['sentiment_score']
        }
    except Exception as e:
        logger.error(f"Error getting S&P 500 data: {e}")
        return {'timestamps': [], 'actual': [], 'predicted': [], 'sentiment_score': []}

@cache_response
def get_nifty_data():
    """Get NIFTY 50 data with caching"""
    try:
        # Return the latest NIFTY 50 data (last 100 points)
        latest_data = nifty_data_store.get_latest_data(100)
        return {
            'timestamps': latest_data['timestamps'],
            'actual': latest_data['actual'],
            'predicted': latest_data['predicted'],
            'sentiment_score': latest_data['sentiment_score']
        }
    except Exception as e:
        logger.error(f"Error getting NIFTY data: {e}")
        return {'timestamps': [], 'actual': [], 'predicted': [], 'sentiment_score': []}

@app.route('/api/sp500', methods=['GET'])
def api_sp500():
    """API endpoint for S&P 500 data"""
    return jsonify(get_sp500_data())

@app.route('/api/nifty', methods=['GET'])
def api_nifty():
    """API endpoint for NIFTY 50 data"""
    return jsonify(get_nifty_data())

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

def cleanup_cache():
    """Periodically clean up expired cache entries"""
    while True:
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, (data, cache_time) in api_cache.items():
                if current_time - cache_time > CACHE_DURATION:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del api_cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error in cache cleanup: {e}")
        
        time.sleep(60)  # Run cleanup every minute

def start_update_threads():
    """Start all background threads"""
    try:
        threading.Thread(target=update_data, daemon=True).start()
        threading.Thread(target=update_nifty_data, daemon=True).start()
        threading.Thread(target=cleanup_cache, daemon=True).start()
        logger.info("All update threads started successfully")
    except Exception as e:
        logger.error(f"Error starting update threads: {e}")

if __name__ == "__main__":
    start_update_threads()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) 