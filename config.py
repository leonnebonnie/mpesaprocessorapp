import os
import json
import logging

logger = logging.getLogger("mpesa_api.config")

# Default configuration
DEFAULT_CONFIG = {
    "input_dir": "Input data",
    "output_dir": "Output data",
    "paybill": "333222",
    "cutoff_time": "16:59:59",
    "previous_closing_balance": None
}

def load_config():
    """
    Load configuration from either:
    1. Environment variables (priority for Azure App Service)
    2. azure_config.json if running in Azure App Service
    3. config.json if exists locally
    4. Default values as fallback
    """
    config = DEFAULT_CONFIG.copy()
    
    # Check if we're running in Azure App Service
    in_azure = os.environ.get('WEBSITE_SITE_NAME') is not None
    
    # Try to load from config file
    config_file = "azure_config.json" if in_azure else "config.json"
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Loaded configuration from {config_file}")
    except Exception as e:
        logger.warning(f"Error loading {config_file}: {str(e)}")
    
    # Override with environment variables if they exist
    if os.environ.get("MPESA_INPUT_DIR"):
        config["input_dir"] = os.environ.get("MPESA_INPUT_DIR")
    
    if os.environ.get("MPESA_OUTPUT_DIR"):
        config["output_dir"] = os.environ.get("MPESA_OUTPUT_DIR")
    
    if os.environ.get("MPESA_PAYBILL"):
        config["paybill"] = os.environ.get("MPESA_PAYBILL")
    
    if os.environ.get("MPESA_CUTOFF_TIME"):
        config["cutoff_time"] = os.environ.get("MPESA_CUTOFF_TIME")
    
    if os.environ.get("MPESA_PREV_BALANCE"):
        try:
            config["previous_closing_balance"] = float(os.environ.get("MPESA_PREV_BALANCE"))
        except (ValueError, TypeError):
            logger.warning(f"Invalid previous balance in environment variable: {os.environ.get('MPESA_PREV_BALANCE')}")
    
    # Ensure directories exist
    os.makedirs(config["input_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    
    return config

# Get the configuration
config = load_config()
