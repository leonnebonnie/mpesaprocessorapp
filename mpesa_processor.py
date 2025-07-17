#!/usr/bin/env python3
"""
Mpesa Statement Processor Application

This application automates the processing of Mpesa statement CSV files:
- Finds and processes all CSV files in a specified directory
- Appends multiple CSV files in chronological order
- Processes the data according to specific business rules
- Splits the processed data into two separate output files based on a time threshold
- Formats the output to match a specified template format
- Validates the data for accuracy and completeness
"""

import os
import re
import csv
import sys
import argparse
import logging
import datetime
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Custom exception for time gaps
class TimeGapException(Exception):
    """Exception raised when time gaps are detected in the transaction data"""
    
    def __init__(self, gaps: List[Dict[str, str]]):
        self.gaps = gaps
        self.message = f"Found {len(gaps)} significant time gaps in the transaction data"
        super().__init__(self.message)
    
    def get_gaps_info(self) -> List[Dict[str, str]]:
        return self.gaps

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mpesa_processor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mpesa_processor")

class MpesaProcessor:
    """Main class for processing Mpesa statements"""
    
    def __init__(self, input_dir: str, output_dir: str, paybill: str, 
                 cutoff_time: str, date: Optional[str] = None, skip_time_gaps: bool = False):
        """
        Initialize the Mpesa processor
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory for output files
            paybill: Paybill number to add to output
            cutoff_time: Time threshold for splitting files (format: HH:MM:SS)
            date: Optional date filter for output filename (format: YYYY-MM-DD)
            skip_time_gaps: Whether to skip time gap validation
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.paybill = paybill
        
        # Ensure cutoff_time is properly formatted as a string (HH:MM:SS)
        self.cutoff_time = str(cutoff_time)  # Convert to string to handle numeric input
        # Validate format - default to 16:59:59 if format is incorrect
        if ':' not in self.cutoff_time:
            logger.warning(f"Invalid cutoff_time format: {cutoff_time}, using default 16:59:59")
            self.cutoff_time = '16:59:59'
            
        self.date = date
        self.statement_date = None
        self.processed_data = None  # Store the processed DataFrame for validation
        self.skip_time_gaps = skip_time_gaps
        self.time_gaps = []  # Store detected time gaps
        
        # Balance tracking
        self.opening_balance = None
        self.closing_balance = None
        self.final_transaction_balance = None  # To store the balance from the last transaction
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validation
        if not self.input_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        logger.info(f"Initialized processor - Input: {input_dir}, Output: {output_dir}, "
                   f"Paybill: {paybill}, Cutoff time: {cutoff_time}, Date: {date}")
    
    def find_csv_files(self) -> List[Path]:
        """
        Find all relevant CSV files in the input directory
        
        Returns:
            List of Path objects for CSV files sorted chronologically
        """
        # Get all CSV files matching the pattern
        pattern = r"ORG_\d+_Utility Account_Completed_(\d{14})\.csv"
        csv_files = []
        
        for file in self.input_dir.glob("*.csv"):
            match = re.match(pattern, file.name)
            if match:
                timestamp = match.group(1)
                csv_files.append((file, timestamp))
        
        # Sort files by timestamp
        csv_files.sort(key=lambda x: x[1])
        
        sorted_files = [file[0] for file in csv_files]
        
        if not sorted_files:
            logger.warning(f"No matching CSV files found in {self.input_dir}")
        else:
            logger.info(f"Found {len(sorted_files)} CSV files")
            for file in sorted_files:
                logger.info(f"  - {file.name}")
        
        return sorted_files
    
    def process_files(self, csv_files: List[Path]) -> pd.DataFrame:
        """
        Process and combine all CSV files
        
        Args:
            csv_files: List of CSV files to process
            
        Returns:
            DataFrame containing combined data that is chronologically sorted
        """
        if not csv_files:
            raise ValueError("No CSV files to process")
        
        all_data = []
        total_records = 0
        opening_balance = None
        closing_balance = None
        self.statement_date = None
        all_datetime_cols = []  # Track datetime columns for sorting
        
        for i, file_path in enumerate(csv_files):
            logger.info(f"Processing file {i+1}/{len(csv_files)}: {file_path.name}")
            
            # Read file with pandas
            try:
                # Skip the metadata rows (first 6 lines)
                df = pd.read_csv(file_path, skiprows=6)
                
                # First file - extract opening balance and statement date from metadata
                if i == 0:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for j, line in enumerate(f):
                            if j == 3:  # 4th line (0-indexed) - Time Period line
                                # Extract statement date
                                # Format: Time Period:,From,12-05-2025 22:40:01,To,12-05-2025 23:59:59
                                match = re.search(r'From,(\d{2}-\d{2}-\d{4})', line)
                                if match:
                                    date_str = match.group(1)
                                    try:
                                        date_obj = datetime.datetime.strptime(date_str, '%d-%m-%Y')
                                        self.statement_date = date_obj.strftime('%Y%m%d')
                                        logger.info(f"Extracted statement date: {self.statement_date}")
                                    except Exception as e:
                                        logger.warning(f"Could not parse statement date: {str(e)}")
                            elif j == 5:  # 6th line (0-indexed)
                                # Extract opening balance
                                match = re.search(r'Opening Balance:,(\d+\.\d+)', line)
                                if match:
                                    opening_balance = float(match.group(1))
                                    logger.info(f"Extracted opening balance: {opening_balance}")
                                break
                
                # Last file - extract closing balance from metadata
                if i == len(csv_files) - 1:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for j, line in enumerate(f):
                            if j == 5:  # 6th line (0-indexed)
                                # Extract closing balance
                                match = re.search(r'Closing Balance:,(\d+\.\d+)', line)
                                if match:
                                    closing_balance = float(match.group(1))
                                    logger.info(f"Extracted closing balance: {closing_balance}")
                                break
                
                total_records += len(df)
                all_data.append(df)
                logger.info(f"Added {len(df)} records from {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                raise
        
        # Combine all dataframes
        logger.info(f"Combining {len(all_data)} dataframes with total {total_records} records")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicate receipts
        try:
            original_count = len(combined_df)
            
            # Identify columns that should uniquely identify a transaction
            # First check if the receipt number column exists (try common variations)
            receipt_columns = [
                'Receipt No.', 'Receipt Number', 'Reference', 'Transaction ID', 'Transaction Reference',
                # Add MPesa specific transaction identifiers
                'Trans ID', 'M-PESA Receipt', 'Receipt', 'Transaction Number', 'Confirmation Number',
                'Mpesa Transaction ID', 'MPesa Receipt', 'Mpesa Receipt', 'Transaction Ref', 'MPESA Receipt'
            ]
            receipt_column = None
            
            for col in receipt_columns:
                if col in combined_df.columns:
                    receipt_column = col
                    break
            
            if receipt_column:
                logger.info(f"Found receipt column: {receipt_column}")
                # Keep the first occurrence of each receipt number (assuming sorted by time)
                # Use multiple columns if possible for better duplicate detection
                duplicate_check_columns = [receipt_column]
                
                # Add Details column for more accurate duplicate detection as specifically requested
                if 'Details' in combined_df.columns:
                    duplicate_check_columns.append('Details')
                    logger.info("Including 'Details' column in duplicate detection")
                
                # Add more columns for more accurate duplicate detection if they exist
                if 'Date' in combined_df.columns:
                    duplicate_check_columns.append('Date')
                if 'Amount' in combined_df.columns:
                    duplicate_check_columns.append('Amount')
                    
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=duplicate_check_columns, keep='first')
                
                # Log the results
                removed_count = original_count - len(combined_df)
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} duplicate receipts based on {duplicate_check_columns}")
                else:
                    logger.info("No duplicate receipts found")
            else:
                logger.warning("Could not find a receipt or reference column for duplicate detection")
        except Exception as e:
            logger.warning(f"Error during duplicate receipt removal: {str(e)}")
            # Continue processing even if duplicate removal fails
        
        # Sort the combined dataframe chronologically
        try:
            logger.info("Sorting combined data chronologically")
            # Convert Date column to datetime if it exists
            if 'Date' in combined_df.columns:
                # Try different date formats
                try:
                    # Try to convert the date column to datetime for proper sorting
                    combined_df['SortDateTime'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')
                    # Sort by the datetime column in descending order (newest to oldest)
                    combined_df = combined_df.sort_values(by='SortDateTime', ascending=False).reset_index(drop=True)
                    # Remove the temporary sorting column
                    combined_df = combined_df.drop('SortDateTime', axis=1)
                    logger.info("Successfully sorted data in reverse chronological order (newest to oldest)")
                except Exception as e:
                    logger.warning(f"Could not sort by Date: {str(e)}")
        except Exception as e:
            logger.warning(f"Error during chronological sorting: {str(e)}")
        
        # Store balances for validation
        self.opening_balance = opening_balance
        self.closing_balance = closing_balance
        
        return combined_df
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data according to requirements
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        logger.info("Transforming data...")
        
        # 1. Fill empty values in Withdrawn and Paid In with 0
        df['Withdrawn'] = df['Withdrawn'].fillna(0)
        df['Paid In'] = df['Paid In'].fillna(0)
        
        # 2. Convert to numeric values to ensure they're treated as numbers
        df['Withdrawn'] = pd.to_numeric(df['Withdrawn'])
        df['Paid In'] = pd.to_numeric(df['Paid In'])
        df['Balance'] = pd.to_numeric(df['Balance'])
        
        # 3. Remove unnecessary columns
        columns_to_remove = ["Initiation Time", "Linked Transaction ID", "A/C No."]
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 4. Rename columns to match output format
        df = df.rename(columns={
            'Receipt No.': 'Receipt', 
            'Completion Time': 'Date',
            'Transaction Status': 'Status',  # Rename to 'Status' as requested
            'Reason Type': 'Transaction Type'  # Rename to 'Transaction Type' as requested
        })
        
        # 5. Add PayBillNumber column
        df['PayBillNumber'] = self.paybill
        
        # 6. Add empty Transaction Party Details and TransactionID columns
        df['Transaction Party Details'] = ''
        df['TransactionID'] = ''
        
        # 7. Rearrange columns to match expected output format
        # The expectation is Withdrawn comes before Paid In
        column_order = [
            'Receipt', 'Date', 'Details', 'Status', 'Withdrawn', 'Paid In',
            'Balance', 'Balance Confirmed', 'Transaction Type', 'Other Party Info',
            'Transaction Party Details', 'TransactionID', 'PayBillNumber'
        ]
        
        # Make sure all required columns exist (some might have different names)
        available_columns = set(df.columns)
        required_columns = set(column_order)
        missing_columns = required_columns - available_columns
        
        if missing_columns:
            # Map column names that might be different
            column_mapping = {
                'Status': 'Status',  # Self-mapping to ensure it exists
                'Transaction Type': 'Transaction Type'  # Self-mapping to ensure it exists
            }
            
            # Apply mapping where needed
            for req_col, avail_col in column_mapping.items():
                if req_col in missing_columns and avail_col in available_columns:
                    df = df.rename(columns={avail_col: req_col})
        
        # Recheck after renames
        available_columns = set(df.columns)
        required_columns = set(column_order)
        missing_columns = required_columns - available_columns
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Add missing columns with empty values
            for col in missing_columns:
                df[col] = ''
        
        # Format date column to DD/MM/YYYY HH:MM:SS (long date format)
        try:
            # Assuming the date is in DD-MM-YYYY HH:MM:SS format
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
            df['Date'] = df['Date'].dt.strftime('%d/%m/%Y %H:%M:%S')
            logger.info("Date column formatted to include seconds (DD/MM/YYYY HH:MM:SS)")
        except Exception as e:
            logger.error(f"Error converting date format: {str(e)}")
            # If conversion fails, leave as is
            
        # Reorder columns to match expected output
        final_columns = [col for col in column_order if col in df.columns]
        df = df[final_columns]
        
        logger.info(f"Transformation complete. Final shape: {df.shape}")
        
        return df
    
    def split_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into two files based on time
        
        Args:
            df: DataFrame to split
            
        Returns:
            Tuple of (late_df, early_df) DataFrames
        """
        logger.info(f"Splitting data by time threshold: {self.cutoff_time}")
        
        # Convert Date back to datetime for comparison
        # First, check if it's already a datetime object
        if not pd.api.types.is_datetime64_dtype(df['Date']):
            try:
                # First try parsing with seconds included (our new format)
                comparison_dates = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M:%S')
                logger.info("Parsed dates with seconds for comparison")
            except Exception as e1:
                try:
                    # Try without seconds (older format)
                    comparison_dates = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
                    logger.info("Parsed dates without seconds for comparison")
                except Exception as e2:
                    try:
                        # Try original input format as last resort
                        comparison_dates = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M:%S')
                        logger.info("Parsed dates in original format for comparison")
                    except Exception as e3:
                        # If all else fails, try flexible parsing
                        logger.warning(f"Failed specific format parsing, using flexible parser: {str(e1)}")
                        comparison_dates = pd.to_datetime(df['Date'], dayfirst=True)
        else:
            comparison_dates = df['Date']
        
        # Extract time component for comparison
        try:
            times = comparison_dates.dt.strftime('%H:%M:%S')
        except Exception as e:
            logger.error(f"Error extracting time components: {str(e)}")
            # Create a safe fallback - all times as midnight
            times = pd.Series(['00:00:00'] * len(df), index=df.index)
        
        # Parse cutoff time - ensure cutoff_time is a string before splitting
        cutoff_time_str = str(self.cutoff_time)  # Convert to string to handle cases where it might be a float
        time_parts = cutoff_time_str.split(':')
        cutoff_hour = int(time_parts[0])
        cutoff_minute = int(time_parts[1])
        cutoff_second = int(time_parts[2]) if len(time_parts) > 2 else 0
        
        # Create filter conditions
        # File 1 (a): 23:59:59 to 16:59:59 (late transactions)
        # File 2 (b): 17:00:00 to 00:00:00 (early transactions)
        
        late_mask = times.apply(lambda t: self._is_late_time(t, cutoff_hour, cutoff_minute, cutoff_second))
        
        # Create copies of the filtered data
        late_df = df[late_mask].copy() if any(late_mask) else df.head(0).copy()  # Empty df with same columns if no late transactions
        early_df = df[~late_mask].copy() if any(~late_mask) else df.head(0).copy()  # Empty df with same columns if no early transactions
        
        # Sort both dataframes by date and balance before any splitting
        try:
            # Create a sorting column in both dataframes
            for sort_df in [late_df, early_df]:
                if not sort_df.empty and 'Date' in sort_df.columns:
                    sort_df['SortDateTime'] = pd.to_datetime(sort_df['Date'], dayfirst=True, errors='coerce')
                    # Convert Balance to numeric for proper sorting
                    if 'Balance' in sort_df.columns:
                        sort_df['BalanceNumeric'] = pd.to_numeric(sort_df['Balance'], errors='coerce')
                        logger.info(f"Converted Balance column to numeric for sorting: {sort_df['BalanceNumeric'].head(3)}")
            
            # Sort both dataframes in reverse chronological order (newest to oldest)
            # AND by Balance in descending order (highest balance first) for same timestamps
            if not late_df.empty and 'SortDateTime' in late_df.columns:
                late_df = late_df.sort_values(
                    by=['SortDateTime', 'BalanceNumeric'], 
                    ascending=[False, False]  # Newest first, Highest balance first
                ).reset_index(drop=True)
                
            if not early_df.empty and 'SortDateTime' in early_df.columns:
                early_df = early_df.sort_values(
                    by=['SortDateTime', 'BalanceNumeric'], 
                    ascending=[False, False]  # Newest first, Highest balance first
                ).reset_index(drop=True)
                
            # Remove temporary sorting columns
            if 'SortDateTime' in late_df.columns:
                late_df = late_df.drop('SortDateTime', axis=1)
            if 'BalanceNumeric' in late_df.columns:
                late_df = late_df.drop('BalanceNumeric', axis=1)
            if 'SortDateTime' in early_df.columns:
                early_df = early_df.drop('SortDateTime', axis=1)
            if 'BalanceNumeric' in early_df.columns:
                early_df = early_df.drop('BalanceNumeric', axis=1)
                
            logger.info("Successfully sorted transactions by time (newest first) and balance (highest first)")
            
            # Log the first few rows to verify sorting
            if not late_df.empty:
                logger.info(f"Late transactions (top 3 rows after sorting):\n{late_df[['Date', 'Balance']].head(3)}")
            if not early_df.empty:
                logger.info(f"Early transactions (top 3 rows after sorting):\n{early_df[['Date', 'Balance']].head(3)}")

        except Exception as e:
            logger.warning(f"Error during chronological sorting of split dataframes: {str(e)}")
        
        # If one of the dataframes is empty but the other has data, split the data
        # to ensure both files have content (50/50 split)
        if len(late_df) > 0 and len(early_df) == 0:
            logger.warning("No early transactions found, splitting late transactions 50/50")
            midpoint = len(late_df) // 2
            early_df = late_df.iloc[midpoint:].copy()
            late_df = late_df.iloc[:midpoint].copy()
        elif len(early_df) > 0 and len(late_df) == 0:
            logger.warning("No late transactions found, splitting early transactions 50/50")
            midpoint = len(early_df) // 2
            late_df = early_df.iloc[midpoint:].copy()
            early_df = early_df.iloc[:midpoint].copy()
        
        logger.info(f"Split complete - Late: {len(late_df)} records, Early: {len(early_df)} records")
        
        return late_df, early_df
    
    def _is_late_time(self, time_str, cutoff_hour: int, cutoff_minute: int, cutoff_second: int) -> bool:
        """
        Check if a time is considered 'late' (after cutoff)
        
        Args:
            time_str: Time string in format HH:MM:SS
            cutoff_hour: Hour component of cutoff
            cutoff_minute: Minute component of cutoff
            cutoff_second: Second component of cutoff
            
        Returns:
            True if time is after cutoff, False otherwise
        """
        try:
            # Ensure time_str is a string
            if not isinstance(time_str, str):
                time_str = str(time_str)
                
            # Handle case where time_str might be a float/numeric value
            if ':' not in time_str:
                logger.warning(f"Invalid time format received: {time_str}, treating as 00:00:00")
                # Default to midnight if the format is invalid
                return False  # Midnight is always "early"
                
            # Now we can safely split
            time_parts = time_str.split(':')
            
            # Handle incomplete time formats
            if len(time_parts) < 2:
                logger.warning(f"Time format missing minutes: {time_str}, treating as 00:00:00")
                return False
                
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(time_parts[2]) if len(time_parts) > 2 else 0
        except (ValueError, IndexError, TypeError) as e:
            # Log the error and default to midnight (00:00:00)
            logger.error(f"Error parsing time '{time_str}': {str(e)}. Treating as 00:00:00")
            return False  # Default to "early" time
        
        # Convert to total seconds for easier comparison
        time_seconds = hour * 3600 + minute * 60 + second
        cutoff_seconds = cutoff_hour * 3600 + cutoff_minute * 60 + cutoff_second
        
        # Special case: if cutoff is in the afternoon and time is in the evening,
        # it's considered "late"
        if cutoff_hour < 12 and hour >= 12:
            return True
        # If time is after cutoff, it's "late"
        elif time_seconds > cutoff_seconds:
            return True
        # Otherwise, it's "early"
        return False
    
    def check_time_gaps(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Check for gaps in the time sequence of transactions
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            List of dictionaries with start and end times of gaps
        """
        # If skip_time_gaps is True, return empty list
        if self.skip_time_gaps:
            logger.info("Skipping time gap detection as requested")
            return []
            
        logger.info("Checking for time gaps in the data...")
        
        # Ensure Date column is datetime
        if not pd.api.types.is_datetime64_dtype(df['Date']):
            try:
                date_col = pd.to_datetime(df['Date'])
            except Exception as e:
                logger.warning(f"Could not convert Date to datetime for gap analysis: {str(e)}")
                return []
        else:
            date_col = df['Date']
        
        # Sort by date in ascending order
        df_sorted = df.copy()
        df_sorted['datetime'] = date_col
        df_sorted = df_sorted.sort_values('datetime', ascending=True)
        
        # Extract transaction date from the first record
        if len(df_sorted) > 0:
            first_date = df_sorted['datetime'].iloc[0].date()
            start_datetime = datetime.datetime.combine(first_date, datetime.time.min)
            end_datetime = datetime.datetime.combine(first_date, datetime.time.max)
            
            # Create a full day time range with 1-minute intervals
            full_day = pd.date_range(start=start_datetime, end=end_datetime, freq='1min')
            
            # Round transaction times to the nearest minute for comparison
            df_sorted['minute'] = df_sorted['datetime'].dt.floor('min')
            transaction_minutes = set(df_sorted['minute'].dt.strftime('%H:%M:%S').tolist())
            
            # Find missing minutes
            all_minutes = set(t.strftime('%H:%M:%S') for t in full_day)
            missing_minutes = all_minutes - transaction_minutes
            
            # Group consecutive missing minutes into ranges
            gaps = []
            if missing_minutes:
                # Convert to datetime.time objects for easier sorting and comparison
                missing_times = sorted([datetime.datetime.strptime(m, '%H:%M:%S').time() for m in missing_minutes])
                
                # Find consecutive ranges
                gap_start = missing_times[0]
                prev_time = gap_start
                
                for i in range(1, len(missing_times)):
                    curr_time = missing_times[i]
                    # Check if times are consecutive (1 minute apart)
                    prev_dt = datetime.datetime.combine(datetime.date.today(), prev_time)
                    curr_dt = datetime.datetime.combine(datetime.date.today(), curr_time)
                    
                    if (curr_dt - prev_dt).total_seconds() > 60:  # If more than 1 minute apart
                        # End the current gap and start a new one
                        gaps.append({
                            'start': gap_start.strftime('%H:%M:%S'),
                            'end': prev_time.strftime('%H:%M:%S'),
                            'duration_minutes': int((datetime.datetime.combine(datetime.date.today(), prev_time) - 
                                              datetime.datetime.combine(datetime.date.today(), gap_start)).total_seconds() / 60) + 1
                        })
                        gap_start = curr_time
                    
                    prev_time = curr_time
                
                # Add the last gap
                gaps.append({
                    'start': gap_start.strftime('%H:%M:%S'),
                    'end': prev_time.strftime('%H:%M:%S'),
                    'duration_minutes': int((datetime.datetime.combine(datetime.date.today(), prev_time) - 
                                      datetime.datetime.combine(datetime.date.today(), gap_start)).total_seconds() / 60) + 1
                })
                
                # Filter out gaps shorter than 3 minutes
                significant_gaps = [gap for gap in gaps if gap['duration_minutes'] >= 3]
                
                if significant_gaps:
                    total_missing_minutes = sum(gap['duration_minutes'] for gap in significant_gaps)
                    logger.warning(f"Found {len(significant_gaps)} significant time gaps with total {total_missing_minutes} minutes missing")
                    for gap in significant_gaps:
                        logger.warning(f"Missing time range: {gap['start']} to {gap['end']} ({gap['duration_minutes']} minutes)")
                else:
                    logger.info("No significant time gaps found")
                    
                return significant_gaps
        
        return []
    
    def get_final_transaction_balance(self, df: pd.DataFrame) -> Optional[float]:
        """
        Extract the balance from the last transaction of the day
        If multiple transactions occur at the same time (e.g., 23:59:59),
        select the one with the highest balance.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Balance from the last transaction of the day or None if not found
        """
        if df.empty or 'Balance' not in df.columns or 'Date' not in df.columns:
            logger.warning("Cannot extract final transaction balance: DataFrame is empty or missing required columns")
            return None
        
        try:
            # Make sure DataFrame is sorted (newest first)
            # We need to make a copy since we will modify it
            sorted_df = df.copy()
            
            # Add a sortable datetime column if not already present
            if not pd.api.types.is_datetime64_dtype(sorted_df.get('Date')):
                try:
                    # Try different date formats for sorting
                    try:
                        # Try with seconds (DD/MM/YYYY HH:MM:SS)
                        sorted_df['SortDateTime'] = pd.to_datetime(sorted_df['Date'], format='%d/%m/%Y %H:%M:%S')
                    except:
                        try:
                            # Try without seconds (DD/MM/YYYY HH:MM)
                            sorted_df['SortDateTime'] = pd.to_datetime(sorted_df['Date'], format='%d/%m/%Y %H:%M')
                        except:
                            try:
                                # Try original format (DD-MM-YYYY HH:MM:SS)
                                sorted_df['SortDateTime'] = pd.to_datetime(sorted_df['Date'], format='%d-%m-%Y %H:%M:%S')
                            except:
                                # Last resort: flexible parsing
                                sorted_df['SortDateTime'] = pd.to_datetime(sorted_df['Date'], dayfirst=True, errors='coerce')
                except Exception as e:
                    logger.error(f"Error creating datetime for sorting: {str(e)}")
                    return None
            else:
                # If it's already a datetime, just copy it
                sorted_df['SortDateTime'] = sorted_df['Date']
            
            # Convert Balance to numeric for comparison and clean up any non-numeric values
            sorted_df['BalanceNumeric'] = pd.to_numeric(sorted_df['Balance'], errors='coerce')
            
            # Drop rows with NaN in either SortDateTime or BalanceNumeric
            valid_data = sorted_df.dropna(subset=['SortDateTime', 'BalanceNumeric'])
            if valid_data.empty:
                logger.warning("No valid datetime and balance pairs found after conversion")
                return None
                
            # First, find the latest time in the dataset
            latest_time = valid_data['SortDateTime'].max()
            logger.info(f"Latest transaction time in dataset: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get all transactions at this exact time
            latest_transactions = valid_data[valid_data['SortDateTime'] == latest_time]
            
            # Debug all transactions at the latest time - useful for troubleshooting
            if len(latest_transactions) > 0:
                for idx, row in latest_transactions.iterrows():
                    receipt = row.get('Receipt', 'N/A')
                    balance = row['BalanceNumeric']
                    date_str = row['Date'] if isinstance(row['Date'], str) else row['SortDateTime'].strftime('%Y-%m-%d %H:%M:%S')
                    logger.debug(f"Transaction at {date_str}: Receipt={receipt}, Balance={balance}")
            
            if len(latest_transactions) > 1:
                logger.info(f"Found {len(latest_transactions)} transactions at the latest time {latest_time.strftime('%H:%M:%S')}")
                
                # From these latest transactions, get the one with the highest balance
                highest_balance_idx = latest_transactions['BalanceNumeric'].idxmax()
                highest_balance_row = latest_transactions.loc[highest_balance_idx]
                final_balance = float(highest_balance_row['BalanceNumeric'])
                
                logger.info(f"Multiple transactions at {latest_time.strftime('%H:%M:%S')}, selected highest balance: {final_balance}")
                logger.info(f"Transaction details: Receipt={highest_balance_row.get('Receipt', 'N/A')}, Balance={final_balance}")
            else:
                # Only one transaction at the latest time
                last_row = latest_transactions.iloc[0]
                final_balance = float(last_row['BalanceNumeric'])
                logger.info(f"Single transaction at latest time {latest_time.strftime('%H:%M:%S')} with balance {final_balance}")
            
            # Double check the final balance value is reasonable and not an error
            if final_balance > 1_000_000_000:  # Sanity check to prevent unrealistic values
                logger.warning(f"Final balance {final_balance} seems unrealistically high, might be an error")
            
            return final_balance
        except Exception as e:
            logger.error(f"Error extracting final transaction balance: {str(e)}")
            return None

    def validate_data(self, combined_df: pd.DataFrame) -> bool:
        """
        Validate data integrity
        
        Args:
            combined_df: Combined DataFrame to validate
            
        Returns:
            True if validation passed, False otherwise
        """
        logger.info("Validating data...")
        
        validation_passed = True
        
        # 1. Check for duplicate transaction IDs
        duplicate_ids = combined_df['Receipt'].duplicated()
        if duplicate_ids.any():
            dup_count = duplicate_ids.sum()
            logger.warning(f"Found {dup_count} duplicate transaction IDs")
            validation_passed = False
        else:
            logger.info("No duplicate transaction IDs found")
        
        # 2. Verify financial integrity
        if self.opening_balance is not None:
            # Calculate the sum of paid in minus withdrawn
            total_paid_in = combined_df['Paid In'].sum()
            total_withdrawn = combined_df['Withdrawn'].sum()
            net_change = total_paid_in - total_withdrawn
            
            # Calculate expected closing balance
            expected_closing = self.opening_balance + net_change
            
            # Get the actual closing balance to use for validation
            # Prioritize the balance from the last transaction if available
            actual_closing = self.final_transaction_balance if self.final_transaction_balance is not None else self.closing_balance
            
            # Log which balance we're using for validation
            if self.final_transaction_balance is not None:
                logger.info(f"Using balance from final transaction for validation: {self.final_transaction_balance}")
                if self.closing_balance is not None:
                    logger.info(f"Metadata closing balance (not used): {self.closing_balance}")
            elif self.closing_balance is not None:
                logger.info(f"Using metadata closing balance for validation: {self.closing_balance}")
            
            # Perform the validation if we have an actual closing balance
            if actual_closing is not None:
                # Compare with actual closing balance (allow small floating point differences)
                if abs(expected_closing - actual_closing) > 0.01:
                    logger.warning(
                        f"Financial integrity check failed: "
                        f"Opening ({self.opening_balance}) + Net Change ({net_change}) = {expected_closing}, "
                        f"but Closing Balance = {actual_closing}"
                    )
                    validation_passed = False
                else:
                    logger.info("Financial integrity check passed")
            else:
                logger.warning("No closing balance available for validation")
        else:
            logger.warning("Could not verify financial integrity - missing opening balance information")
        
        # 3. Check for time sequence gaps
        self.time_gaps = self.check_time_gaps(combined_df)
        if self.time_gaps and not self.skip_time_gaps:
            logger.warning(f"Found {len(self.time_gaps)} significant time gaps")
            validation_passed = False
        
        if validation_passed:
            logger.info("All validation checks passed")
        else:
            logger.warning("Some validation checks failed - see log for details")
        
        return validation_passed
    
    def save_output(self, late_df: pd.DataFrame, early_df: pd.DataFrame) -> Tuple[str, str]:
        """
        Save the split data to output files
        
        Args:
            late_df: DataFrame for late transactions (File 1)
            early_df: DataFrame for early transactions (File 2)
            
        Returns:
            Tuple of (file1_path, file2_path)
        """
        # Determine output date string
        date_str = None
        
        # Option 1: If data exists, extract date from the first entry
        if not late_df.empty or not early_df.empty:
            # Get the non-empty dataframe
            df_to_use = late_df if not late_df.empty else early_df
            
            # Extract date from the first transaction
            try:
                # Try to extract from the Date field (should be in DD/MM/YYYY HH:MM or DD-MM-YYYY HH:MM:SS format)
                first_date = df_to_use['Date'].iloc[0]
                
                # Parse the date
                if isinstance(first_date, str):
                    try:
                        if '/' in first_date:
                            # Format: DD/MM/YYYY HH:MM
                            date_parts = first_date.split()[0].split('/')
                            # Check if year is 2-digit or 4-digit
                            year = date_parts[2]
                            if len(year) == 2:
                                year = f"20{year}"
                            date_str = f"{year}{date_parts[1]}{date_parts[0]}"
                        elif '-' in first_date:
                            # Format: DD-MM-YYYY HH:MM:SS
                            date_obj = datetime.datetime.strptime(first_date.split()[0], '%d-%m-%Y')
                            date_str = date_obj.strftime('%Y%m%d')
                    except Exception as e:
                        logger.warning(f"Error parsing date string '{first_date}': {str(e)}")
                        date_str = None
                elif hasattr(first_date, 'strftime'):
                    # If it's a datetime-like object, format it directly
                    date_str = first_date.strftime('%Y%m%d')
                else:
                    logger.warning(f"Unrecognized date format: {type(first_date)} - {first_date}")
                    date_str = None
                    
                logger.info(f"Extracted date {date_str} from transaction data")
            except Exception as e:
                logger.warning(f"Could not extract date from transaction data: {str(e)}")
                date_str = None
        
        # Option 2: Use the provided date parameter
        if date_str is None and self.date:
            date_obj = datetime.datetime.strptime(self.date, '%Y-%m-%d')
            date_str = date_obj.strftime('%Y%m%d')
            logger.info(f"Using provided date parameter: {date_str}")
        
        # Option 3: Default to current date
        if date_str is None:
            date_str = datetime.datetime.now().strftime('%Y%m%d')
            logger.info(f"Using current date: {date_str}")
        
        # Early transactions should have suffix 'a'
        # Late transactions should have suffix 'b'
        early_file_name = f"{date_str} Incoming Mpesa Statement a.xlsx"
        late_file_name = f"{date_str} Incoming Mpesa Statement b.xlsx"
        
        early_file_path = self.output_dir / early_file_name
        late_file_path = self.output_dir / late_file_name
        
        logger.info(f"Named output files: early='{early_file_name}', late='{late_file_name}'")
        
        # Try to save files, handle permission errors
        try:
            # Save late transactions to 'b' file
            logger.info(f"Saving late transactions to {late_file_path}")
            late_df.to_excel(late_file_path, index=False, engine='openpyxl')
            logger.info(f"Saved {len(late_df)} late transactions to {late_file_path}")
            
            # Save early transactions to 'a' file
            logger.info(f"Saving early transactions to {early_file_path}")
            early_df.to_excel(early_file_path, index=False, engine='openpyxl')
            logger.info(f"Saved {len(early_df)} early transactions to {early_file_path}")
        except PermissionError as e:
            # If files are locked, try with timestamp in filename
            timestamp = datetime.datetime.now().strftime('%H%M%S')
            
            # Create new filenames with timestamps
            early_file_name = f"{date_str} Incoming Mpesa Statement a_{timestamp}.xlsx"  # Early = 'a'
            late_file_name = f"{date_str} Incoming Mpesa Statement b_{timestamp}.xlsx"   # Late = 'b'
            
            early_file_path = self.output_dir / early_file_name
            late_file_path = self.output_dir / late_file_name
            
            logger.warning(f"Permission error on original files, trying with timestamp: early={early_file_path}, late={late_file_path}")
            
            # Try again with new filenames
            late_df.to_excel(late_file_path, index=False, engine='openpyxl')
            early_df.to_excel(early_file_path, index=False, engine='openpyxl')
            logger.info(f"Saved files with timestamp: early={early_file_path}, late={late_file_path}")
        except Exception as e:
            logger.error(f"Error saving output files: {str(e)}")
            raise
        
        # Return paths as strings
        # Note: We return late_file_path first to maintain compatibility with existing code
        # This is because late_file_path corresponds to file1_path in the original code
        return str(late_file_path), str(early_file_path)
        
    def process(self, skip_time_gaps: bool = False) -> Tuple[str, str]:
        """
        Process all files and generate output
        
        Args:
            skip_time_gaps: Whether to proceed despite time gaps
            
        Returns:
            Tuple of (file1_path, file2_path)
        """
        logger.info("Starting Mpesa statement processing")
        
        # Update skip_time_gaps flag if passed
        if skip_time_gaps:
            self.skip_time_gaps = True
        
        # 1. Find CSV files
        csv_files = self.find_csv_files()
        
        # 2. Process all files
        combined_df = self.process_files(csv_files)
        
        # Transform data
        transformed_df = self.transform_data(combined_df)
        
        # Store the processed data for validation
        self.processed_data = transformed_df.copy()
        
        # Get the final transaction balance (from the transaction closest to 23:59:59)
        self.final_transaction_balance = self.get_final_transaction_balance(transformed_df)
        
        # If final_transaction_balance was found, log it for comparison
        if self.final_transaction_balance is not None:
            logger.info(f"Extracted final balance from last transaction: {self.final_transaction_balance}")
            if self.closing_balance is not None:
                logger.info(f"Compared to extracted metadata closing balance: {self.closing_balance}")
                if abs(self.final_transaction_balance - self.closing_balance) > 0.01:
                    logger.warning(f"Discrepancy between final transaction balance and metadata closing balance!")
                else:
                    logger.info(f"Final transaction balance matches metadata closing balance")
        
        # Validate data
        validation_result = self.validate_data(transformed_df)
        
        # If there are time gaps and we're not skipping them, raise an exception
        if self.time_gaps and not self.skip_time_gaps:
            raise TimeGapException(self.time_gaps)
        
        # Split data by time
        late_df, early_df = self.split_by_time(transformed_df)
        
        # Store the late and early dataframes as instance attributes so they can be accessed later
        self.late_df = late_df.copy()
        self.early_df = early_df.copy()
        
        # Log information about the split data
        logger.info(f"Late transactions: {len(late_df)} rows")
        if not late_df.empty and 'Balance' in late_df.columns:
            try:
                # Get the last transaction in chronological order in the late DataFrame
                # Need to sort the late_df by date again to ensure we get the latest transaction
                sorted_late_df = late_df.copy()
                sorted_late_df['SortDateTime'] = pd.to_datetime(sorted_late_df['Date'], dayfirst=True, errors='coerce')
                sorted_late_df = sorted_late_df.sort_values('SortDateTime', ascending=True).reset_index(drop=True)
                
                if not sorted_late_df.empty:
                    last_late_txn = sorted_late_df.iloc[-1]
                    last_time = last_late_txn['SortDateTime'].strftime('%H:%M:%S') if 'SortDateTime' in last_late_txn else 'N/A'
                    logger.info(f"Last late transaction at {last_time}: Receipt={last_late_txn.get('Receipt', 'N/A')}, Balance={last_late_txn.get('Balance', 'N/A')}")
            except Exception as e:
                logger.error(f"Error accessing last late transaction: {str(e)}")
        
        # 6. Save output
        file1_path, file2_path = self.save_output(late_df, early_df)
        
        logger.info("Processing completed successfully")
        
        return file1_path, file2_path

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Process Mpesa statements")
    
    parser.add_argument("--input-dir", required=True, help="Directory containing input CSV files")
    parser.add_argument("--output-dir", required=True, help="Directory for output files")
    parser.add_argument("--paybill", required=True, help="Paybill number to add to output")
    parser.add_argument("--cutoff-time", required=True, help="Time threshold for splitting files (format: HH:MM:SS)")
    parser.add_argument("--date", help="Optional date for output filename (format: YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    try:
        processor = MpesaProcessor(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            paybill=args.paybill,
            cutoff_time=args.cutoff_time,
            date=args.date
        )
        
        file1_path, file2_path = processor.process()
        
        print(f"\nProcessing complete!")
        print(f"Output files:")
        print(f"  - {file1_path}")
        print(f"  - {file2_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
