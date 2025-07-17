#!/usr/bin/env python3
import os
import sys

print("Starting Mpesa Statement Processor...")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add current directory to Python path
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Import your Flask app
    from app import app
    print("Successfully imported Flask app")
    
    # Get port from environment variable (Azure sets this)
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting app on port: {port}")
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
    
except Exception as e:
    print(f"Error starting application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
