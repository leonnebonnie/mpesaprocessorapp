from flask import Flask, request, jsonify, send_from_directory, Response
import pandas as pd
from flask_cors import CORS
import os
import sys
import json
import logging
import threading
import time
from pathlib import Path
from mpesa_processor import MpesaProcessor, TimeGapException
from datetime import datetime
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mpesa_api")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for all routes

# Create directories for static files and templates
os.makedirs(os.path.join(os.getcwd(), "static"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "templates"), exist_ok=True)

# Store the processing status
processing_status = {
    "is_processing": False,
    "progress": 0,
    "message": "",
    "last_update": None,
    "error": None,
    "output_files": [],
    "validation_result": None,
    "financial_summary": None,
    "time_gaps": None,
    "awaiting_time_gap_decision": False,
    "processor_params": None
}

# Ensure required directories exist
os.makedirs(os.path.join(os.getcwd(), "Input data"), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), "Output data"), exist_ok=True)

def reset_status():
    """Reset the processing status to default values"""
    global processing_status
    processing_status = {
        "is_processing": False,
        "progress": 0,
        "message": "",
        "last_update": None,
        "error": None,
        "output_files": [],
        "validation_result": None,
        "financial_summary": None,
        "time_gaps": None,
        "awaiting_time_gap_decision": False,
        "processor_params": None
    }

def update_status(progress, message, error=None, output_files=None, validation_result=None, financial_summary=None, time_gaps=None, awaiting_decision=None):
    """Update the processing status"""
    global processing_status
    processing_status["progress"] = progress
    processing_status["message"] = message
    processing_status["last_update"] = datetime.now().isoformat()
    
    if error:
        processing_status["error"] = str(error)
    
    if output_files:
        processing_status["output_files"] = output_files
        
    if validation_result is not None:
        processing_status["validation_result"] = validation_result
        
    if financial_summary is not None:
        processing_status["financial_summary"] = financial_summary
        
    if time_gaps is not None:
        processing_status["time_gaps"] = time_gaps
        
    if awaiting_decision is not None:
        processing_status["awaiting_time_gap_decision"] = awaiting_decision

# Create a logger wrapper that updates processing status
class StatusLogger(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        
        # Update progress based on log message patterns
        progress = processing_status["progress"]
        
        # Check for specific patterns to update progress
        if "Initialized processor" in msg:
            progress = 10
        elif "Found" in msg and "CSV files" in msg:
            progress = 20
        elif "Processing file" in msg:
            # Extract progress from "Processing file X/Y"
            try:
                parts = msg.split("Processing file ")[1].split("/")
                current = int(parts[0])
                total = int(parts[1].split(":")[0])
                progress = 20 + min(50, (current / total) * 50)  # 20-70% for file processing
            except:
                progress = 30
        elif "Combining" in msg:
            progress = 70
        elif "Transforming data" in msg:
            progress = 75
        elif "Validating data" in msg:
            progress = 80
        elif "Splitting data" in msg:
            progress = 85
        elif "Saving" in msg:
            progress = 90
        elif "Processing completed successfully" in msg:
            progress = 100
        
        # Update status
        update_status(progress, msg)

# Add the status logger to mpesa_processor's logger
status_handler = StatusLogger()
status_handler.setLevel(logging.INFO)
logging.getLogger("mpesa_processor").addHandler(status_handler)

def process_files_thread(input_dir, output_dir, paybill, cutoff_time, date=None, previous_closing_balance=None, skip_time_gaps=False):
    """Run the Mpesa processor in a separate thread"""
    global processing_status
    processing_status["is_processing"] = True
    
    try:
        # Store parameters for possible reuse
        if not skip_time_gaps:
            processing_status["processor_params"] = {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "paybill": paybill,
                "cutoff_time": cutoff_time,
                "date": date,
                "previous_closing_balance": previous_closing_balance
            }
        
        # Run the processor
        processor = MpesaProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            paybill=paybill,
            cutoff_time=cutoff_time,
            date=date,
            skip_time_gaps=skip_time_gaps
        )
        
        try:
            # Process the files and get the paths to the output files
            # file1_path is for late transactions (b file)
            # file2_path is for early transactions (a file)
            file1_path, file2_path = processor.process()
            
            # Capture the late_df and early_df from the processor
            # The late_df contains the transactions we want for the final balance
            late_df = processor.late_df  # This is a new attribute we'll add to MpesaProcessor
            
            # Get relative paths for the frontend
            rel_file1 = os.path.relpath(file1_path, os.getcwd())
            rel_file2 = os.path.relpath(file2_path, os.getcwd())
            
            # Store just the file names for simplicity
            output_files = [
                os.path.basename(file1_path),  # Late transactions (b file)
                os.path.basename(file2_path)   # Early transactions (a file)
            ]
            
            logger.info(f"Late transactions file: {os.path.basename(file1_path)}")
            logger.info(f"Early transactions file: {os.path.basename(file2_path)}")
            
            # If previous closing balance was provided, perform validation
            validation_result = None
            financial_summary = None
            
            if previous_closing_balance is not None and processor.processed_data is not None:
                try:
                    prev_balance = float(previous_closing_balance)
                    total_paid_in = float(processor.processed_data['Paid In'].sum())
                    total_withdrawn = float(processor.processed_data['Withdrawn'].sum())
                    
                    # Debug the input values for balance calculation
                    logger.info(f"Raw values for balance calculation:")
                    logger.info(f"  - Previous closing balance: {prev_balance}")
                    logger.info(f"  - Total paid in: {total_paid_in}")
                    logger.info(f"  - Total withdrawn: {total_withdrawn}")
                    
                    # Calculate the final balance - Withdrawn is already negative, so we add it
                    calculated_final_balance = prev_balance + total_paid_in + total_withdrawn
                    logger.info(f"Balance calculation: {prev_balance} + {total_paid_in} + {total_withdrawn} = {calculated_final_balance}")
                    
                    # If the calculated balance is significantly different from actual, add more analysis
                    if not processor.processed_data.empty and 'Balance' in processor.processed_data.columns:
                        try:
                            # Get first and last transactions for debugging
                            first_txn = processor.processed_data.iloc[0]
                            last_txn = processor.processed_data.iloc[-1]
                            
                            logger.info(f"First transaction in data: Receipt={first_txn.get('Receipt', 'N/A')}, Balance={first_txn.get('Balance', 'N/A')}")
                            logger.info(f"Last transaction in data: Receipt={last_txn.get('Receipt', 'N/A')}, Balance={last_txn.get('Balance', 'N/A')}")
                            
                            # The calculation may be wrong if prev_balance is incorrect
                            # Instead, check if we can calculate directly from the data
                            if len(processor.processed_data) > 1:
                                first_balance = float(processor.processed_data['Balance'].iloc[0])
                                txn_difference = total_paid_in + total_withdrawn
                                logger.info(f"Alternative calculation: First balance {first_balance} + Transaction difference {txn_difference} = {first_balance + txn_difference}")
                        except Exception as e:
                            logger.error(f"Error during balance analysis: {str(e)}")
                    
                    # Regardless of calculated value, use the actual final balance from the data 
                    # for comparison to maintain data integrity
                    
                    # We need to get the balance from the late transactions (b file), not from the combined data
                    # The late transactions file contains the last transaction of the day which has the final balance
                    if hasattr(processor, 'late_df') and not processor.late_df.empty and 'Balance' in processor.late_df.columns:
                        # Get balances from the late transactions file
                        late_balances = processor.late_df['Balance'].tolist()
                        logger.info(f"Balance values in late transactions: {late_balances[-5:]} (last 5)")
                        
                        # The correct actual final balance should be the balance from the last row of the late transactions file
                        # This is the last transaction chronologically in the late file
                        if 'Receipt' in processor.late_df.columns:
                            # Print receipt/balance pairs from late transactions for debugging
                            late_receipts = processor.late_df['Receipt'].tolist()
                            logger.info(f"Late transactions receipt/balance pairs: {list(zip(late_receipts, late_balances))}")
                            
                            # Log the last transaction from late file which is the one we want
                            last_late_txn = processor.late_df.iloc[-1]
                            logger.info(f"Using last transaction from late file: Receipt={last_late_txn.get('Receipt')}, Balance={last_late_txn.get('Balance')}")
                    
                    # Still get combined data balances for diagnostics
                    if not processor.processed_data.empty and 'Balance' in processor.processed_data.columns:
                        # Get balances from all data for debugging
                        all_balances = processor.processed_data['Balance'].tolist()
                        logger.info(f"All balance values: {all_balances[-5:]} (last 5)")
                        
                        if 'Receipt' in processor.processed_data.columns:
                            # Print all receipt/balance pairs for debugging
                            all_receipts = processor.processed_data['Receipt'].tolist()
                            logger.info(f"All receipt/balance pairs: {list(zip(all_receipts[-10:], all_balances[-10:]))} (last 10)")

                        
                        # CRITICAL: We need the actual final balance from the LAST TRANSACTION IN THE LATE FILE
                        # The late transactions file (b file) has the transactions that happen after the cutoff time
                        # and we need to use the last transaction from there for the actual final balance
                        
                        # Initialize the actual_final_balance to None
                        actual_final_balance = None
                        
                        # Use the balance from the last transaction in the late transactions file
                        if hasattr(processor, 'late_df') and not processor.late_df.empty and 'Balance' in processor.late_df.columns:
                            try:
                                # Use processor's final_transaction_balance if available (highest balance at latest time)
                                if hasattr(processor, 'final_transaction_balance') and processor.final_transaction_balance is not None:
                                    actual_final_balance = processor.final_transaction_balance
                                    logger.info(f"FOUND FINAL BALANCE: {actual_final_balance} from final_transaction_balance")
                                else:
                                    # Create a copy for sorting to get the latest transactions by time
                                    late_df_copy = processor.late_df.copy()
                                    
                                    # Add datetime column for proper sorting
                                    try:
                                        # Convert Date column to datetime using various formats
                                        try:
                                            late_df_copy['SortDateTime'] = pd.to_datetime(late_df_copy['Date'], format='%d/%m/%Y %H:%M:%S')
                                        except:
                                            try:
                                                late_df_copy['SortDateTime'] = pd.to_datetime(late_df_copy['Date'], format='%d/%m/%Y %H:%M')
                                            except:
                                                try:
                                                    late_df_copy['SortDateTime'] = pd.to_datetime(late_df_copy['Date'], format='%d-%m-%Y %H:%M:%S')
                                                except:
                                                    late_df_copy['SortDateTime'] = pd.to_datetime(late_df_copy['Date'], dayfirst=True, errors='coerce')
                                        
                                        # Convert Balance to numeric for proper comparison
                                        late_df_copy['BalanceNumeric'] = pd.to_numeric(late_df_copy['Balance'], errors='coerce')
                                        
                                        # Find the latest timestamp in the data
                                        latest_time = late_df_copy['SortDateTime'].max()
                                        logger.info(f"Latest transaction time: {latest_time.strftime('%H:%M:%S')}")
                                        
                                        # Get all transactions at this latest time
                                        latest_transactions = late_df_copy[late_df_copy['SortDateTime'] == latest_time]
                                        
                                        # If multiple transactions at the latest time, get the one with highest balance
                                        if len(latest_transactions) > 1:
                                            logger.info(f"Found {len(latest_transactions)} transactions at latest time {latest_time.strftime('%H:%M:%S')}")
                                            # Get the transaction with the highest balance
                                            highest_balance_row = latest_transactions.loc[latest_transactions['BalanceNumeric'].idxmax()]
                                            actual_final_balance = float(highest_balance_row['BalanceNumeric'])
                                            last_txn_receipt = highest_balance_row.get('Receipt', 'Unknown')
                                            logger.info(f"FOUND FINAL BALANCE: {actual_final_balance} from highest balance transaction at latest time ({last_txn_receipt})")
                                        else:
                                            # Just one transaction at the latest time
                                            last_row = latest_transactions.iloc[0]
                                            actual_final_balance = float(last_row['BalanceNumeric'])
                                            last_txn_receipt = last_row.get('Receipt', 'Unknown')
                                            logger.info(f"FOUND FINAL BALANCE: {actual_final_balance} from only transaction at latest time ({last_txn_receipt})")
                                    except Exception as e:
                                        logger.error(f"Error during datetime sorting: {e}")
                                        # Fallback to simpler method if datetime approach fails
                                        last_late_txn = processor.late_df.iloc[-1]
                                        actual_final_balance = float(last_late_txn['Balance'])
                                        last_txn_receipt = last_late_txn.get('Receipt', 'Unknown')
                                        logger.info(f"FALLBACK FINAL BALANCE: {actual_final_balance} from last transaction in late file ({last_txn_receipt})")
                            except Exception as e:
                                logger.error(f"Error getting final balance from late transactions: {e}")
                        
                        # If we couldn't get the balance from the late transactions file, fall back to the original method
                        if actual_final_balance is None:
                            logger.warning("Couldn't get final balance from late transactions file, falling back to sorted method")
                            sorted_time = processor.processed_data.copy()
                            try:
                                # Convert the date column to proper datetime for accurate sorting
                                sorted_time['DateTimeObj'] = pd.to_datetime(sorted_time['Date'], dayfirst=True, errors='coerce')
                                # Sort by the datetime to get the newest transaction
                                sorted_time = sorted_time.sort_values('DateTimeObj', ascending=False)
                                # The first row is now the latest transaction
                                actual_final_balance = float(sorted_time['Balance'].iloc[0])
                                last_txn_receipt = sorted_time['Receipt'].iloc[0] if 'Receipt' in sorted_time.columns else 'Unknown'
                                logger.info(f"Final balance {actual_final_balance} from newest transaction {last_txn_receipt} (fallback method)")
                            except Exception as e:
                                # Fallback to last row if datetime sorting fails
                                logger.error(f"Error sorting by datetime: {e}")
                                actual_final_balance = float(processor.processed_data['Balance'].iloc[-1])
                                logger.info(f"Fallback to last row balance: {actual_final_balance}")
                                
                        # Ensure we have a valid actual_final_balance
                        if actual_final_balance is None:
                            logger.error("No valid final balance found! Using calculated balance instead.")
                            actual_final_balance = calculated_final_balance
                        
                        # Use the actual final balance from the newest transaction without any hardcoding
                        # This ensures we always use what's in the statement data
                        logger.info(f"Using actual final balance: {actual_final_balance}")
                        # Don't force any specific balance value, just use what we find in the data
                        
                        # Clear and detailed validation for: Prev balance + Total paid in + Total withdrawn = Final balance
                        # The formula should match exactly for financial integrity, but we'll allow a small tolerance
                        # for potential rounding errors or timestamp differences
                        
                        # Re-calculate expected final balance with a clear formula
                        expected_final = prev_balance + total_paid_in + total_withdrawn
                        
                        # Ensure we're using our calculated value consistently
                        calculated_final_balance = expected_final
                        
                        # Detailed logging for transparency
                        logger.info(f"VALIDATION FORMULA: Previous Closing Balance + Total Paid In + Total Withdrawn = Expected Final Balance")
                        logger.info(f"VALIDATION VALUES: {prev_balance:,.2f} + {total_paid_in:,.2f} + {total_withdrawn:,.2f} = {expected_final:,.2f}")
                        logger.info(f"ACTUAL BALANCE FROM FINAL TRANSACTION: {actual_final_balance:,.2f}")
                        
                        # Calculate absolute and percentage differences
                        difference = calculated_final_balance - actual_final_balance
                        abs_difference = abs(difference)
                        percentage_diff = (abs_difference / actual_final_balance) * 100 if actual_final_balance != 0 else 0
                        
                        # Set tolerance to exactly 0.00% as requested - requiring exact match
                        base_tolerance_amount = 0.0  # No KES difference allowed
                        base_tolerance_percentage = 0.0  # 0% tolerance - must match exactly
                        
                        # Set the tolerance to exactly zero for all dataset sizes
                        txn_count = len(processor.processed_data)
                        adjusted_tolerance_percentage = 0.0
                        
                        # Final tolerance is zero - requiring exact match
                        adjusted_tolerance = 0.0
                        
                        # Log tolerance information
                        logger.info(f"Base tolerance: {base_tolerance_percentage}% or {base_tolerance_amount} KES")
                        logger.info(f"Adjusted tolerance for {txn_count} transactions: {adjusted_tolerance_percentage}%")
                        logger.info(f"Final tolerance amount: {adjusted_tolerance:,.2f} KES")
                        logger.info(f"Actual difference: {abs_difference:,.2f} KES ({percentage_diff:.3f}%)")
                        
                        # Determine if difference is within acceptable tolerance
                        is_valid = abs_difference <= adjusted_tolerance
                        
                        # Format values for display
                        difference = calculated_final_balance - actual_final_balance  # Can be negative
                        percentage_diff = (abs(difference) / actual_final_balance) * 100 if actual_final_balance != 0 else 0
                        
                        # Create a detailed user-friendly status message with clear explanation
                        if is_valid:
                            status_message = "✅ Perfect Match"
                        else:
                            status_message = f"❌ Invalid (Difference of {abs_difference:,.2f} KES - Expected exact match)"
                        
                        validation_result = {
                            "Validation Status": status_message,
                            "Previous Day's Closing Balance": f"KES {prev_balance:,.2f}",
                            "Total Paid In": f"KES {total_paid_in:,.2f}",
                            "Total Withdrawn": f"KES {total_withdrawn:,.2f}",
                            "Expected Final Balance": f"KES {calculated_final_balance:,.2f}",
                            "Actual Final Balance": f"KES {actual_final_balance:,.2f}",
                            "Difference": f"KES {difference:,.2f} ({percentage_diff:.2f}%)"
                        }
                        
                        # Log the entire balance column and receipts for diagnostics
                        logger.info(f"All receipts: {processor.processed_data['Receipt'].tolist()}")
                        logger.info(f"All balances: {processor.processed_data['Balance'].tolist()}")
                        
                        # Get the first row (index 0) since data is in reverse chronological order (newest first)
                        # Due to our sorting, index 0 is the newest/latest transaction
                        last_row_balance = float(processor.processed_data['Balance'].iloc[0])
                        last_receipt = processor.processed_data['Receipt'].iloc[0]
                        logger.info(f"Using balance {last_row_balance} from newest transaction {last_receipt}")
                        
                        # Also log what we would get from the last row for comparison
                        last_index_balance = float(processor.processed_data['Balance'].iloc[-1])
                        last_index_receipt = processor.processed_data['Receipt'].iloc[-1]
                        logger.info(f"Last index would give balance {last_index_balance} from receipt {last_index_receipt}")
                        # Debug log the actual final balance we'll use
                        logger.info(f"===== FINAL BALANCE SELECTION: {actual_final_balance} =====")
                        
                        # Generate the results to show
                        financial_summary = {
                            "Previous Closing Balance": f"KES {prev_balance:,.2f}",
                            "Total Paid In": f"KES {total_paid_in:,.2f}",
                            "Total Withdrawn": f"KES {total_withdrawn:,.2f}",
                            "Final Balance": f"KES {actual_final_balance:,.2f}",
                            "Total Transaction Count": f"{len(processor.processed_data):,}"
                        }
                        
                        logger.info(f"Financial summary using final balance: KES {actual_final_balance:,.2f}")
                        
                        logger.info(f"Financial summary - final balance: KES {last_row_balance:,.2f}")
                        
                        # Log for debugging
                        logger.info(f"Validation result: {validation_result}")
                        logger.info(f"Financial summary: {financial_summary}")
                        
                        logger.info(f"Financial validation: {is_valid}, Calculated: {calculated_final_balance}, Actual: {actual_final_balance}")
                except Exception as e:
                    logger.error(f"Error during financial validation: {str(e)}")
            
            # Update final status - explicitly setting is_processing to False
            processing_status["is_processing"] = False
            
            update_status(
                100,
                "Processing completed successfully",
                output_files=output_files,
                validation_result=validation_result,
                financial_summary=financial_summary,
                awaiting_decision=False
            )
            
        except TimeGapException as e:
            # Handle time gaps
            time_gaps = e.get_gaps_info()
            logger.warning(f"Time gaps detected: {time_gaps}")
            
            # Update status to show time gaps and wait for user decision
            update_status(
                50,  # Set progress to 50% when waiting for user decision
                "Time gaps detected in transaction data. Waiting for user decision to continue or cancel.",
                time_gaps=time_gaps,
                awaiting_decision=True
            )
            processing_status["is_processing"] = False  # Pause processing until user decision
    
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        update_status(0, "Error processing files", error=str(e), awaiting_decision=False)
        processing_status["is_processing"] = False
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        update_status(0, "Error processing files", error=str(e))
    
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        update_status(0, "Error processing files", error=str(e), awaiting_decision=False)
        processing_status["is_processing"] = False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current processing status"""
    return jsonify(processing_status)

@app.route('/api/process', methods=['POST'])
def process_files():
    """Start processing the Mpesa files"""
    if processing_status["is_processing"]:
        return jsonify({"error": "A processing job is already running"}), 400
    
    # If we're waiting for a time gap decision, don't allow starting a new process
    if processing_status["awaiting_time_gap_decision"]:
        return jsonify({"error": "Waiting for decision on time gaps. Please continue or cancel the current process."}), 400
    
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ["inputDir", "outputDir", "paybill", "cutoffTime"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Reset status
        reset_status()
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=process_files_thread,
            args=(
                data["inputDir"],
                data["outputDir"],
                data["paybill"],
                data["cutoffTime"],
                data.get("date"),  # Optional field
                data.get("previousClosingBalance"),  # Optional field for validation
                False  # skip_time_gaps is False for initial processing
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Processing started"}), 202
    
    except Exception as e:
        logger.error(f"Error starting process: {str(e)}")

@app.route('/api/files/input', methods=['GET'])
def list_input_files():
    """List all files in the input directory"""
    input_dir = config["input_dir"]  # Get input directory from config
    try:
        path = Path(input_dir)
        
        if not path.exists() or not path.is_dir():
            return jsonify({"error": f"Directory not found: {input_dir}"}), 404
        
        files = []
        for file_path in path.glob("*.csv"):
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
        
        return jsonify({"files": files})
    
    except Exception as e:
        logger.error(f"Error listing input files: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/output', methods=['GET'])
def list_output_files():
    """List all files in the output directory"""
    output_dir = config["output_dir"]  # Get output directory from config
    try:
        path = Path(output_dir)
        
        if not path.exists() or not path.is_dir():
            return jsonify({"error": f"Directory not found: {output_dir}"}), 404
        
        files = []
        for file_path in path.glob("*.*"):  # Include both .xlsx and .csv
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
        
        return jsonify({"files": files})
    
    except Exception as e:
        logger.error(f"Error listing output files: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download a file"""
    try:
        # Determine the directory
        input_dir = config['input_dir']
        output_dir = config['output_dir']
        
        if filename.startswith(f'{input_dir}/'):
            directory = input_dir
            filename = filename.replace(f'{input_dir}/', '')
        elif filename.startswith(f'{output_dir}/'):
            directory = output_dir
            filename = filename.replace(f'{output_dir}/', '')
        else:
            return jsonify({"error": "Invalid file path"}), 400
        
        return send_from_directory(directory, filename, as_attachment=True)
    
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/continue-with-gaps', methods=['POST'])
def continue_with_gaps():
    """Continue processing despite time gaps"""
    global processing_status
    
    if not processing_status["awaiting_time_gap_decision"]:
        return jsonify({"error": "No pending time gap decision"}), 400
    
    try:
        # Get the saved parameters
        params = processing_status["processor_params"]
        if not params:
            return jsonify({"error": "Missing processor parameters"}), 500
        
        # Update status
        processing_status["awaiting_time_gap_decision"] = False
        processing_status["is_processing"] = True
        update_status(60, "Continuing processing despite time gaps...")
        
        # Start processing in a separate thread with skip_time_gaps=True
        thread = threading.Thread(
            target=process_files_thread,
            args=(
                params["input_dir"],
                params["output_dir"],
                params["paybill"],
                params["cutoff_time"],
                params.get("date"),
                params.get("previous_closing_balance"),
                True  # skip_time_gaps is True to continue processing
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"message": "Processing continued"}), 202
    
    except Exception as e:
        logger.error(f"Error continuing process: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cancel-processing', methods=['POST'])
def cancel_processing():
    """Cancel processing due to time gaps"""
    global processing_status
    
    if not processing_status["awaiting_time_gap_decision"] and not processing_status["is_processing"]:
        return jsonify({"error": "No active processing to cancel"}), 400
    
    # Reset status
    reset_status()
    update_status(0, "Processing cancelled by user")
    
    return jsonify({"message": "Processing cancelled"}), 200

# Serve Flask HTML page
@app.route('/')
def index():
    from flask import render_template
    
    # Create basic HTML template if it doesn't exist
    template_path = os.path.join(os.getcwd(), "templates", "index.html")
    if not os.path.exists(template_path):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mpesa Statement Processor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .main-container { max-width: 1200px; margin: 0 auto; }
        .banner { background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; }
        .file-list { height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 4px; }
        .file-list-item { padding: 5px; border-bottom: 1px solid #eee; }
        .file-list-item:last-child { border-bottom: none; }
        .progress-container { margin: 15px 0; }
        .status-message { font-weight: bold; margin: 10px 0; }
        .validation-container { background-color: #f0f8ff; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .financial-summary { background-color: #f0fff0; padding: 15px; border-radius: 5px; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="banner">
            <h1 class="text-center">Mpesa Statement Processor</h1>
            <p class="text-center">Upload and process Mpesa statements with time gap detection</p>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Configuration</h5>
                    </div>
                    <div class="card-body">
                        <form id="processingForm">
                            <div class="mb-3">
                                <label for="inputDir" class="form-label">Input Directory</label>
                                <input type="text" class="form-control" id="inputDir" value="Input data" required>
                            </div>
                            <div class="mb-3">
                                <label for="outputDir" class="form-label">Output Directory</label>
                                <input type="text" class="form-control" id="outputDir" value="Output data" required>
                            </div>
                            <div class="mb-3">
                                <label for="paybill" class="form-label">Paybill Number</label>
                                <input type="text" class="form-control" id="paybill" value="333222" required>
                            </div>
                            <div class="mb-3">
                                <label for="cutoffTime" class="form-label">Cutoff Time</label>
                                <input type="text" class="form-control" id="cutoffTime" value="16:59:59" required>
                            </div>
                            <div class="mb-3">
                                <label for="date" class="form-label">Date (optional)</label>
                                <input type="date" class="form-control" id="date">
                            </div>
                            <div class="mb-3">
                                <label for="prevBalance" class="form-label">Previous Day's Closing Balance</label>
                                <div class="input-group">
                                    <span class="input-group-text">KES</span>
                                    <input type="text" class="form-control" id="prevBalance" placeholder="Enter amount (e.g. 10,750.00)">
                                </div>
                                <div class="form-text">Enter the closing balance from the previous day. Both formats like 10,750.00 and 10750.00 are accepted.</div>
                            </div>
                            
                            <div id="buttonContainer" class="w-100">
                                <button type="submit" class="btn btn-primary w-100" id="startButton">Start Processing</button>
                            </div>
                        </form>
                    </div>
                </div>
                
                <div class="card mt-3">
                    <div class="card-header">
                        <h5>Input Files</h5>
                    </div>
                    <div class="card-body">
                        <div class="file-list" id="inputFiles"></div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Processing Status</h5>
                    </div>
                    <div class="card-body">
                        <div class="progress-container">
                            <div class="progress">
                                <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%;"></div>
                            </div>
                        </div>
                        
                        <div id="statusMessage" class="status-message">Ready to process files...</div>
                        
                        <div id="timeGapsAlert" class="alert alert-warning mt-3 d-none">
                            <h5><i class="fas fa-exclamation-triangle me-2"></i>Missing Time Ranges Detected</h5>
                            <p>The following time ranges are missing from the transaction data. This may indicate incomplete data.</p>
                            
                            <div class="table-responsive mt-2">
                                <table class="table table-bordered table-sm">
                                    <thead>
                                        <tr>
                                            <th>#</th>
                                            <th>Start Time</th>
                                            <th>End Time</th>
                                            <th>Duration</th>
                                        </tr>
                                    </thead>
                                    <tbody id="timeGapsTable"></tbody>
                                </table>
                            </div>
                            
                            <p class="mb-0 mt-3">Would you like to continue processing despite these gaps, or cancel the operation?</p>
                            
                            <div class="mt-3">
                                <button id="continueButton" class="btn btn-success me-2">Continue Processing</button>
                                <button id="cancelButton" class="btn btn-danger">Cancel Processing</button>
                            </div>
                        </div>
                        
                        <div id="resultContainer" class="d-none mt-4">
                            <h5>Output Files</h5>
                            <div class="file-list" id="outputFiles"></div>
                        </div>
                        
                        <div id="validationContainer" class="validation-container d-none mt-3">
                            <h5>Validation Results</h5>
                            <div id="validationResults"></div>
                        </div>
                        
                        <div id="financialContainer" class="financial-summary d-none mt-3">
                            <h5>Financial Summary</h5>
                            <div id="financialResults"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // API URL
        const API_URL = '';
        
        // Check status interval
        let statusInterval;
        
        // Function to update UI based on status
        function updateUI(data) {
            // Update progress
            $('#progressBar').css('width', data.progress + '%').text(data.progress + '%');
            if (data.progress === 100) {
                $('#progressBar').removeClass('bg-primary bg-warning').addClass('bg-success');
            } else if (data.awaiting_time_gap_decision) {
                $('#progressBar').removeClass('bg-primary bg-success').addClass('bg-warning');
            } else {
                $('#progressBar').removeClass('bg-success bg-warning').addClass('bg-primary');
            }
            
            // Update status message
            if (data.message) {
                $('#statusMessage').text(data.message);
            }
            
            // Handle time gaps
            if (data.time_gaps && data.time_gaps.length > 0 && data.awaiting_time_gap_decision) {
                $('#timeGapsAlert').removeClass('d-none');
                let tableHtml = '';
                data.time_gaps.forEach((gap, index) => {
                    tableHtml += `
                        <tr>
                            <td>${index + 1}</td>
                            <td>${gap.start}</td>
                            <td>${gap.end}</td>
                            <td>${gap.duration_minutes} minutes</td>
                        </tr>
                    `;
                });
                $('#timeGapsTable').html(tableHtml);
            } else {
                $('#timeGapsAlert').addClass('d-none');
            }
            
            // Handle output files
            if (data.output_files && data.output_files.length > 0) {
                $('#resultContainer').removeClass('d-none');
                let filesHtml = '';
                data.output_files.forEach(file => {
                    // Handle both string and object formats
                    let fileName = typeof file === 'string' ? file : (file.name || file.toString());
                    filesHtml += `
                        <div class="file-list-item">
                            <i class="fas fa-file me-2"></i>
                            <a href="${API_URL}/download/${fileName}" target="_blank">${fileName}</a>
                        </div>
                    `;
                });
                $('#outputFiles').html(filesHtml);
            }
            
            // Handle validation results
            if (data.validation_result) {
                $('#validationContainer').removeClass('d-none');
                let validationHtml = '<ul class="list-group">';
                for (const [key, value] of Object.entries(data.validation_result)) {
                    validationHtml += `
                        <li class="list-group-item">
                            <strong>${key}:</strong> ${value}
                        </li>
                    `;
                }
                validationHtml += '</ul>';
                $('#validationResults').html(validationHtml);
            }
            
            // Handle financial summary
            if (data.financial_summary) {
                $('#financialContainer').removeClass('d-none');
                let financialHtml = '<ul class="list-group">';
                for (const [key, value] of Object.entries(data.financial_summary)) {
                    financialHtml += `
                        <li class="list-group-item">
                            <strong>${key}:</strong> ${value}
                        </li>
                    `;
                }
                financialHtml += '</ul>';
                $('#financialResults').html(financialHtml);
            }
            
            // Update button state
            if (data.is_processing || data.awaiting_time_gap_decision) {
                $('#startButton').prop('disabled', true);
                if (data.is_processing) {
                    $('#startButton').html(`
                        <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                        <span>Processing...</span>
                    `);
                }
            } else {
                // Reset button state when processing is done
                $('#startButton').prop('disabled', false).text('Start Processing');
            }
            
            // Special case: if progress is 100%, force button to reset regardless of other states
            if (data.progress === 100) {
                $('#startButton').prop('disabled', false).text('Start Processing');
            }
            
            // Also check for completed processing by looking for results
            if (data.progress === 100 || data.validation_result || data.financial_summary) {
                // Force reset the button state when we have results regardless of is_processing flag
                $('#startButton').prop('disabled', false).text('Start Processing');
                
                // Always reset the processing status when we have results
                fetch(API_URL + '/reset-processing-status', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log('Processing status reset successfully');
                })
                .catch(error => console.error('Error resetting processing status:', error));
                
                // Stop checking status since processing is complete
                clearInterval(statusInterval);
            }
            
            // Handle errors
            if (data.error) {
                alert('Error: ' + data.error);
            }
            
            // If processing is done or awaiting decision, stop the interval
            if (!data.is_processing && data.progress === 100 || data.awaiting_time_gap_decision) {
                clearInterval(statusInterval);
            }
        }
        
        // Check status
        function checkStatus() {
            fetch(API_URL + '/status')
                .then(response => response.json())
                .then(data => {
                    updateUI(data);
                })
                .catch(error => {
                    console.error('Error checking status:', error);
                });
        }
        
        // Load input files
        function loadInputFiles() {
            fetch(API_URL + '/input-files')
                .then(response => response.json())
                .then(data => {
                    if (data.files && data.files.length > 0) {
                        let filesHtml = '';
                        data.files.forEach(file => {
                            filesHtml += `<div class="file-list-item">${file}</div>`;
                        });
                        $('#inputFiles').html(filesHtml);
                    } else {
                        $('#inputFiles').html('<div class="text-center">No input files found</div>');
                    }
                })
                .catch(error => {
                    console.error('Error loading input files:', error);
                });
        }
        
        // Start processing
        function startProcessing(e) {
            e.preventDefault();
            
            // Clear previous results
            $('#resultContainer').addClass('d-none');
            $('#validationContainer').addClass('d-none');
            $('#financialContainer').addClass('d-none');
            $('#timeGapsAlert').addClass('d-none');
            
            // Get form values
            const inputDir = $('#inputDir').val();
            const outputDir = $('#outputDir').val();
            const paybill = $('#paybill').val();
            const cutoffTime = $('#cutoffTime').val();
            const date = $('#date').val();
            const prevBalance = $('#prevBalance').val();
            
            // Create request data
            const requestData = {
                input_dir: inputDir,
                output_dir: outputDir,
                paybill: paybill,
                cutoff_time: cutoffTime
            };
            
            if (date) {
                requestData.date = date;
            }
            
            if (prevBalance) {
                requestData.previous_closing_balance = parseFloat(prevBalance);
            }
            
            // Send request
            fetch(API_URL + '/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Start checking status
                statusInterval = setInterval(checkStatus, 1000);
            })
            .catch(error => {
                console.error('Error starting process:', error);
                alert('Failed to start processing: ' + error.message);
            });
        }
        
        // Continue with gaps
        function continueWithGaps() {
            fetch(API_URL + '/continue-with-gaps', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                $('#timeGapsAlert').addClass('d-none');
                statusInterval = setInterval(checkStatus, 1000);
            })
            .catch(error => {
                console.error('Error continuing process:', error);
                alert('Failed to continue processing: ' + error.message);
            });
        }
        
        // Cancel processing
        function cancelProcessing() {
            fetch(API_URL + '/cancel-processing', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                $('#timeGapsAlert').addClass('d-none');
                $('#startButton').prop('disabled', false).text('Start Processing');
            })
            .catch(error => {
                console.error('Error cancelling process:', error);
                alert('Failed to cancel processing: ' + error.message);
            });
        }
        
        // Initialize
        $(document).ready(function() {
            // Load input files
            loadInputFiles();
            
            // Check initial status
            checkStatus();
            
            // Form submit event
            $('#processingForm').on('submit', startProcessing);
            
            // Time gap decision buttons
            $('#continueButton').on('click', continueWithGaps);
            $('#cancelButton').on('click', cancelProcessing);
        });
    </script>
</body>
</html>
"""
        with open(template_path, 'w') as f:
            f.write(html_content)
    
    # Get input files list
    input_dir = os.path.join(os.getcwd(), "Input data")
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".xlsx") or f.endswith(".xls")]
    
    return render_template('index.html')

# API Endpoints
@app.route('/status')
def status():
    # Log what's in the status for debugging
    logger.debug(f"Status endpoint - validation_result: {processing_status.get('validation_result')}")
    logger.debug(f"Status endpoint - financial_summary: {processing_status.get('financial_summary')}")
    return jsonify(processing_status)
    
@app.route('/input-files')
def input_files():
    """List all files in the input directory"""
    try:
        input_dir = os.path.join(os.getcwd(), "Input data")
        
        if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
            return jsonify({"files": []})
        
        files = []
        for file in os.listdir(input_dir):
            if file.endswith(".xlsx") or file.endswith(".xls") or file.endswith(".csv"):
                file_path = os.path.join(input_dir, file)
                files.append(file)
        
        return jsonify({"files": files})
    except Exception as e:
        logger.error(f"Error listing input files: {str(e)}")
        return jsonify({"files": []})

@app.route('/process', methods=['POST'])
def process_api():
    try:
        if processing_status["is_processing"]:
            return jsonify({"error": "A processing job is already running"}), 400
        
        # If we're waiting for a time gap decision, don't allow starting a new process
        if processing_status["awaiting_time_gap_decision"]:
            return jsonify({"error": "Waiting for decision on time gaps. Please continue or cancel the current process."}), 400
        
        data = request.json
        
        # Reset status
        reset_status()
        
        # Extract field values from the request data (with defaults from config)
        input_dir = data.get('input_dir', config['input_dir'])
        output_dir = data.get('output_dir', config['output_dir'])
        paybill = data.get('paybill', config['paybill'])
        
        # Ensure cutoff_time is properly formatted as a string (HH:MM:SS)
        cutoff_time = str(data.get('cutoff_time', config['cutoff_time']))
        # Validate cutoff_time format
        if ':' not in cutoff_time:
            cutoff_time = '16:59:59'  # Default if format is incorrect
        
        date = data.get('date')
        previous_balance = None
        
        # Convert previous_closing_balance to float if it exists
        if 'previous_closing_balance' in data and data['previous_closing_balance'] is not None:
            balance_str = str(data['previous_closing_balance']).strip()
            
            # Remove currency symbols, commas and extra spaces if present
            balance_str = balance_str.replace('KES', '').replace('Ksh', '').replace('ksh', '')
            balance_str = balance_str.replace(',', '').replace('₹', '').replace('$', '').strip()
            
            # Handle both period and comma as decimal separators
            if '.' in balance_str:
                # Using period as decimal separator
                try:
                    previous_balance = float(balance_str)
                    logger.info(f"Converted previous closing balance: {balance_str} to {previous_balance}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid previous closing balance format (period): {data.get('previous_closing_balance')}")
            elif ',' in balance_str:  # Handle European format with comma decimal separator
                try:
                    # Replace comma with period for float conversion
                    previous_balance = float(balance_str.replace(',', '.'))
                    logger.info(f"Converted previous closing balance: {balance_str} to {previous_balance}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid previous closing balance format (comma): {data.get('previous_closing_balance')}")
            else:
                # Just a plain number
                try:
                    previous_balance = float(balance_str)
                    logger.info(f"Converted previous closing balance: {balance_str} to {previous_balance}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid previous closing balance: {data.get('previous_closing_balance')}")
                    
            # Validate the value makes sense (not negative or extremely large)
            if previous_balance is not None:
                if previous_balance < 0:
                    logger.warning(f"Negative previous closing balance detected: {previous_balance}")
                elif previous_balance > 10000000000:  # Sanity check for extremely large values
                    logger.warning(f"Unusually large previous closing balance detected: {previous_balance}")
                    # Continue processing but warn the user
        
        # Start processing in a separate thread
        thread = threading.Thread(
            target=process_files_thread,
            args=(
                input_dir,
                output_dir,
                paybill,
                cutoff_time,
                date,
                previous_balance,
                False  # skip_time_gaps is False for initial processing
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"success": True, "message": "Processing started"}), 200
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<path:filename>')
def download_api(filename):
    return download_file(filename)

@app.route('/continue-with-gaps', methods=['POST'])
def continue_with_gaps_api():
    return continue_with_gaps()

@app.route('/cancel-processing', methods=['POST'])
def cancel_processing_api():
    return cancel_processing()

@app.route('/reset-processing-status', methods=['POST'])
def reset_processing_status_api():
    """Reset the processing status when the UI detects completion"""
    global processing_status
    processing_status["is_processing"] = False
    logger.info("Processing status manually reset from UI")
    return jsonify({"success": True, "message": "Processing status reset"}), 200

if __name__ == '__main__':
    # Get port from environment variable or default to 5000 (for Azure App Service)
    port = int(os.environ.get('PORT', 5000))
    
    # In Azure App Service, we need to listen on 0.0.0.0
    # For local development, use 127.0.0.1 (localhost)
    host = '0.0.0.0' if os.environ.get('WEBSITE_SITE_NAME') else '127.0.0.1'
    
    app.run(host=host, port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')
    if __name__ == '__main__':
    import os
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)
