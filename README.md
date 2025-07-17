# Mpesa Statement Processor

A Python application that automates the processing of Mpesa statement CSV files exported from the Mpesa portal. This application streamlines the financial reconciliation process by parsing, validating, and transforming transaction data according to specific business rules.

## Portable Application Setup

To run this application on any computer:

1. **Transfer the Application**:
   - Copy the entire `Mpesaauto` folder to the target computer
   - No installation is needed - all files must stay together

2. **Setup Requirements**:
   - Ensure Python 3.7+ is installed on the target computer
   - **Important**: Visual Studio Build Tools are required for some dependencies
      - The application will automatically attempt to download and install these if missing
      - If installation fails, see the "Prerequisites" section below
   - Run the included `setup.bat` script to install all dependencies

3. **Start the Application**:
   - Double-click on `start_app.bat` to launch the application manager
   - Select option 1 to start the application
   - Select option 4 (or respond 'Y' when prompted) to open in your browser

4. **Using the Application**:
   - Put your Mpesa statement files in the `Input data` folder
   - Processed files will appear in the `Output data` folder

## Key Features

### Data Processing
- Automatically finds and processes CSV files exported from the Mpesa portal
- Intelligently combines multiple CSV files in chronological order
- Detects and removes duplicate transactions based on receipt numbers
- Handles both Excel and CSV input file formats
- Sorts transactions in reverse chronological order (newest transactions first)

### Time-Based Processing
- Splits transactions into two separate files based on a configurable cutoff time (default: 16:59:59)
- Ensures both output files contain data through intelligent splitting if all transactions fall on one side of the cutoff
- Generates files with appropriate naming convention: 
  - `YYYYMMDD Incoming Mpesa Statement a.xlsx` (early transactions) 
  - `YYYYMMDD Incoming Mpesa Statement b.xlsx` (late transactions)

### Data Validation
- Performs precise financial integrity validation:
  - Validates that Previous Day's Closing Balance + Total Paid In + Total Withdrawn = Final Balance (with 0.00% tolerance)
- Detects time gaps in transaction data for complete coverage verification
- Identifies significant gaps (3+ minutes) in the transaction timeline
- Provides user options to continue or cancel processing when gaps are found

### Output Formatting
- Formats date/time in complete format (DD/MM/YYYY HH:MM:SS)
- Uses consistent column headers: "Status" instead of "Transaction Status" and "Transaction Type" instead of "Reason Type"
- Extracts and uses transaction date from input files for output filenames
- Saves output in Excel (.xlsx) format for better compatibility

### Robust Error Handling
- Provides automatic recovery for locked output files
- Implements multiple date parsing fallbacks for handling various format variations
- Generates detailed logs for troubleshooting and auditing

## Prerequisites

### Visual Studio Build Tools

This application requires Visual Studio Build Tools because some Python packages (like pandas and numpy) have C extensions that need to be compiled during installation. You have two options:

#### Option 1: Automatic Installation (Recommended)

The included `setup.bat` script will attempt to automatically download and install VS Build Tools if they're missing.

#### Option 2: Manual Installation

If the automatic installation fails:

1. Download Visual Studio Build Tools from [Microsoft's website](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During installation, select the "Desktop development with C++" workload
3. Install the package

#### Option 3: Pre-compiled Wheels (No Build Tools Required)

If you cannot install VS Build Tools, you can install pre-compiled wheel packages instead:

```
pip install --only-binary=:all: -r requirements.txt
```

This will avoid compilation by using pre-built packages where available.

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```
pip install -r requirements.txt
```

Or run the included `setup.bat` script which handles both dependencies and prerequisites.

## Usage

### Using the Web Interface (Recommended)

1. **Launch the Application**:
   - Double-click on `start_app.bat`
   - Select option 1 to start the application
   - Select option 4 to open the web interface in your browser

2. **Process Files**:
   - In the web interface, configure your processing options:
     - Input Directory: Where your Mpesa statement files are located (default: "Input data")
     - Output Directory: Where processed files will be saved (default: "Output data")
     - Paybill: The Mpesa paybill number (default: 333222)
     - Cutoff Time: Time threshold for splitting files (format: HH:MM:SS, default: 16:59:59)
     - Date: Optional date for output filename (format: YYYY-MM-DD)
     - Previous Day's Closing Balance: The closing balance from the previous day, required for financial validation
   - Click "Start Processing" to begin
   - The application will show progress and results in the interface

3. **Time Gap Detection**:
   If significant gaps (3+ minutes) in the transaction timeline are detected:
   - The application will display a list of all detected gaps with:
     - Start time
     - End time
     - Duration (in minutes)
   - You can choose to:
     - Continue processing despite the gaps
     - Cancel the processing operation

4. **Financial Validation**:
   - The application validates that: Previous Day's Closing Balance + Total Paid In + Total Withdrawn = Final Balance
   - The final balance is taken from the last transaction of the day (23:59:59) with the highest balance
   - Validation results show:
     - Previous Day's Closing Balance (from user input)
     - Total Paid In and Total Withdrawn (calculated from transactions)
     - Expected Final Balance (calculated)
     - Actual Final Balance (from last transaction)
     - Difference and validation status

5. **Managing the Application**:
   - Use the `start_app.bat` menu to:
     - Start the application (Option 1)
     - Restart the application (Option 2)
     - Stop the application (Option 3)
     - Open the browser interface (Option 4)
     - Exit the application manager (Option 5)

### Command Line Usage (Advanced)

You can also run the application directly using the command line:

```
python app.py
```

This starts the web server, which you can access at http://localhost:5000

## Output Files

The application generates two Excel output files:

1. `YYYYMMDD Incoming Mpesa Statement a.xlsx`: Transactions with completion time before the cutoff time (early transactions)
2. `YYYYMMDD Incoming Mpesa Statement b.xlsx`: Transactions with completion time after the cutoff time (late transactions)

### File Features

- Both files are in Excel (.xlsx) format for better compatibility
- Transaction date is formatted as DD/MM/YYYY HH:MM:SS with full timestamp information
- Transactions are sorted newest-to-oldest (reverse chronological order) for easier review
- Duplicate receipts are automatically detected and removed
- If all transactions would fall into one file, the data is intelligently split to ensure both files contain data
- Column headers use standardized terminology ("Status" and "Transaction Type")
- Each file includes the following columns:
  - Receipt
  - Date (full timestamp)
  - Details
  - Status
  - Withdrawn
  - Paid In
  - Balance
  - Balance Confirmed
  - Transaction Type
  - Other Party Info
  - Transaction Party Details
  - TransactionID
  - PayBillNumber

### Filename Format

The date in the filename (`YYYYMMDD`) is determined in the following order:

1. Extracted from the statement date in the input files (preferred)
2. Provided via the date parameter if extraction fails
3. Current date as fallback if neither of the above is available

### Automatic Recovery

If output files are locked or can't be written to (e.g., open in Excel), the application automatically generates filenames with timestamps to avoid access issues.

## Logging

Logs are written to `mpesa_processor.log` in the current directory.
