# Imaging Overview Report
This applicaton takes the imaging data and provides a visual overview of the Report.  

## Data
The data comes from the Imaging and QC output files from Redcap. These files are loaded using the internal datastore network.

## Data Processing - Analysis
The Imaging and QC input data are modified / analyzed to create the following data products The functions outlining these steps are in the data_processing.py file.

### Basic Outputs:
- Completions: Take a set of imaging data and determine which percent for each scan type were completed
- Roll up: Count of image by site and visit
- Indicated-Received: data comes in with scans as columns. Lengthens the dataframe to create a scan column with the particular scan as a variable, while preserving the indicated / received columns
- Ratings: Merges the indicated-received table with the ratings values from the qc table

### Logic rules used in the Analysis
Overdue:
- Use 'surgery_week' column.
- For Visit 1: overdue if the surgery week is earlier than today
- For Visit 3: overdue if today is >90 days after the surgery_week date
