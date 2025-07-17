import os
import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print(f"Environment variables: PORT={os.environ.get('PORT')}")

try:
    import flask
    print(f"Flask version: {flask.__version__}")
except ImportError as e:
    print(f"Flask import error: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Pandas import error: {e}")

# Simple Flask app
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Azure! Python is working."

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
