  # Libraries
# Data
# File Management
import os # Operating system library
import pathlib # file paths
import json
import requests
import math
import numpy as np
import pandas as pd # Dataframe manipulations
import datetime
from datetime import date
from datetime import datetime, timedelta
from config_settings import *


# ----------------------------------------------------------------------------
# Imaging Data Cleanup
# ----------------------------------------------------------------------------

def clean_imaging(imaging_full):
    ''' Clean up the incoming imaging dataframe'''
    # Imaging columns actually used.  Subset to just these portion of the data. 
    # Dictionary keys = columns used, dictionary value = new column name
    imaging_columns_dict = {
        'site': 'site',
        'subject_id': 'subject_id',
        'visit': 'visit',
        'acquisition_week': 'acquisition_week',
        'Surgery Week':'Surgery Week',
        'bids':'bids',
        'dicom':'dicom', 
        'T1 Indicated':'T1',
        'DWI Indicated':'DWI',
        '1st Resting State Indicated':'REST1',
        'fMRI Individualized Pressure Indicated':'CUFF1',
        'fMRI Standard Pressure Indicated':'CUFF2',
        '2nd Resting State Indicated':'REST2',
        'T1 Received':'T1 Received',
        'DWI Received':'DWI Received',
        '1st Resting State Received':'REST1 Received',
        'fMRI Individualized Pressure Received':'CUFF1 Received',
        'fMRI Standard Pressure Received':'CUFF2 Received',
        '2nd Resting State Received':'REST2 Received',
        'Cuff1 Applied Pressure':'Cuff1 Applied Pressure'
}
    
    imaging_cols = list(imaging_columns_dict.keys()) # Get list of columns to keep
    imaging = imaging_full[imaging_cols].copy() # Copy subset of imaging dataframe
    imaging.rename(columns=imaging_columns_dict, inplace=True) # Rename columns
    imaging = imaging.replace('na', np.nan) # Replace 'na' string with actual null value
    imaging['completions_id'] = imaging.apply(lambda x: str(x['subject_id']) + x['visit'],axis=1, result_type='reduce') # Add completions id value from combination of subject ID and visit
    
    return imaging

def clean_qc(qc_full):
    ''' Clean up the incoming qc dataframe'''
    qc_columns_dict = {
            'site':'site', 
            'sub': 'subject_id',
            'ses': 'ses',
            'scan':'scan',
            'rating': 'rating'
        }
    qc_cols = list(qc_columns_dict.keys()) # Get list of columns to keep
    qc = qc_full[qc_cols].copy() # Copy subset of imaging dataframe
    qc.rename(columns=qc_columns_dict, inplace=True) # Rename columns
    
    return qc

def generate_missing_qc(imaging, qc):
    # Get complete list of scans that would be expected
    scan_types = ['CUFF1', 'CUFF2', 'DWI', 'REST1', 'REST2', 'T1w']
    scan_types_df = pd.DataFrame(scan_types, columns=['scan'])

    # Select needed cols from imaging and rename cols
    imaging_cols = ['site','subject_id', 'visit']
    imaging_qc = imaging[imaging_cols].copy()
    imaging_qc.columns = ['site', 'sub', 'ses']

    # Outer cross imaging with scan types for full expected list
    imaging_qc = imaging_qc.merge(scan_types_df, how='cross')

    # Merge with ratings data from qc
    full_ratings = imaging_qc.merge(qc, on=['site', 'sub', 'ses','scan'], how='outer')

    # Fill NaN ratings withs 'unavailable'
    full_ratings.fillna({"rating": "unavailable"}, inplace = True)
    
    return full_ratings

# ----------------------------------------------------------------------------
# Filter imaging by date
# ----------------------------------------------------------------------------
def relative_date(nDays):
    today = datetime.today()
    relativeDate = (today - pd.Timedelta(days=nDays)).date()
    return relativeDate

def filter_imaging_by_date(imaging_df, start_date = None, end_date = None):
    '''Filter the imaging datatable using:
    start_date: select imaging records acquired on or after this date
    end_date: select imaging records acquired on or before this date'''
    filtered_imaging = imaging_df.copy()
    # filtered_imaging['acquisition_week']= pd.to_datetime(filtered_imaging['acquisition_week'], errors = 'coerce')
    filtered_imaging['acquisition_week'] = pd.to_datetime(filtered_imaging['acquisition_week']).dt.date
    
    if start_date and isinstance(start_date, date):
        filtered_imaging = filtered_imaging[filtered_imaging['acquisition_week'] >= start_date]

    if end_date and isinstance(end_date, date):
        filtered_imaging = filtered_imaging[filtered_imaging['acquisition_week'] <= end_date]

    return filtered_imaging


# ----------------------------------------------------------------------------
# Filter imaging by data release
# ----------------------------------------------------------------------------

def filter_by_release(imaging, release_list):
    ''' Filter imaging list to only include the V1 visit for subjects from specific releases. '''
    filtered_imaging = imaging.copy()
    filtered_imaging = filtered_imaging[(filtered_imaging['subject_id'].isin(release_list)) & (filtered_imaging['visit']=='V1') ]
    
    return filtered_imaging

# ----------------------------------------------------------------------------
# Filter qc by filtered imaging
# ----------------------------------------------------------------------------

def filter_qc(qc, filtered_imaging):
    '''Filter qc records to just those subjects / visits in the filtered imaging set'''
    filtered_qc = qc.copy()
    filtered_qc.loc[:,'ses'] = filtered_qc['ses'].astype('category')
    
    filt_sub = filtered_imaging[['subject_id','visit']].copy()
    filt_sub.columns = ['sub','ses']    
    filt_sub.loc[:,'ses'] = filt_sub['ses'].astype('category')
    
    filtered_qc = qc.merge(filt_sub, how = 'inner',on = ['sub','ses'])
    return filtered_qc

# ----------------------------------------------------------------------------
# Discrepancies Analysis
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Discrepancies Analysis
# ----------------------------------------------------------------------------

def calculate_overdue(BIDS, visit, surgery_week):
    today = datetime.now().date()
    if surgery_week is pd.NaT:
        overdue='No Surgery Date'
    elif BIDS == 0 and visit == 'V1' and surgery_week < today:
        overdue = 'Yes'
    elif BIDS == 0 and visit == 'V3' and  (today-surgery_week).days > 90:
        overdue = 'Yes'
    else:
        overdue='No'
    return overdue

def get_indicated_received(imaging_dataframe, validation_column = 'bids', validation_value = 1):
    """The get_indicated_received(imaging_dataframe) function takes the imaging log data frame and lengthens the
    table to convert the scan into a variable while preserving columns for the indicated and received value of each scan.
    Validation columns parameter should be a lits of tuples where the first tuple value is the column name and the
    second entry is the value of 'Y' for that column"""
    df = clean_imaging(imaging_dataframe).copy()

    # Select columns, and create long dataframes from those columns, pivoting the scan into a variable
    index_cols = ['site','subject_id','visit','acquisition_week','Surgery Week','bids', 'dicom']
    index_new = ['Site', 'Subject', 'Visit','Acquisition Week','Surgery Week', 'BIDS','DICOM']

    # Select and pivot indicated columns
    indicated_cols = ['T1', 'DWI', 'REST1', 'CUFF1', 'CUFF2', 'REST2']

    indicated = df[index_cols + indicated_cols]
    indicated = pd.melt(indicated, id_vars=index_cols, value_vars = indicated_cols)
    indicated.columns = index_new + ['Scan', 'Value']

    # Select and pivot received_cols columns, renaming the scans so they match
    received_cols = ['T1 Received', 'DWI Received', 'REST1 Received', 'CUFF1 Received',
           'CUFF2 Received', 'REST2 Received']

    received = df[index_cols + received_cols]
    received.columns = index_cols + indicated_cols
    received = pd.melt(received, id_vars=index_cols, value_vars = indicated_cols)
    received.columns = index_new + ['Scan', 'Value']

    # Merge the indicated and received dataframes into a single dataframe
    merge_on = index_new + ['Scan']
    combined = pd.merge(indicated, received, how='outer', on=index_new + ['Scan'] )
    combined.columns = index_new + ['Scan','Indicated','Received']

    # Convert columns to dates and calculate if overdue
    combined['Surgery Week'] = pd.to_datetime(combined['Surgery Week'], errors='coerce').dt.date
    combined['Acquisition Week'] = pd.to_datetime(combined['Acquisition Week'], errors='coerce').dt.date
    combined['Overdue'] = combined.apply(lambda x: calculate_overdue(x['BIDS'], x['Visit'], x['Surgery Week']), axis=1)

    return combined


# ----------------------------------------------------------------------------
# Imaging Overview
# ----------------------------------------------------------------------------
def roll_up(imaging):
    print(imaging.columns)
    print(len(imaging))
    cols = ['site','visit','subject_id']
    df = imaging[cols].copy()
    df = df.groupby(['site','visit']).count().reset_index()
    df = df.pivot(index='site', columns = 'visit', values = 'subject_id')
    if len(df >0):
        df.loc['All Sites'] = df.sum(numeric_only=True, axis=0)
    df.loc[:,'Total'] = df.sum(numeric_only=True, axis=1)
    df.reset_index(inplace=True)
    return df

# ----------------------------------------------------------------------------
# Completions
# ----------------------------------------------------------------------------
def get_completions(df):
    scan_dict = {'T1 Indicated':'T1',
       'DWI Indicated':'DWI',
       '1st Resting State Indicated':'REST1',
       'fMRI Individualized Pressure Indicated':'CUFF1',
       'fMRI Standard Pressure Indicated':'CUFF2',
       '2nd Resting State Indicated':'REST2'}

    icols = list(scan_dict.keys())
    icols2 = list(scan_dict.values())

    # df['completions_id'] = df.apply(lambda x: str(x['subject_id']) + x['visit'],axis=1)
    completions = df[['completions_id']+icols].groupby(icols).count().reset_index().rename(columns=scan_dict).rename(columns={'completions_id':'Count'})
    completions['Percent'] = round(100 * completions['Count']/(completions['Count'].sum()),1)
    completions = completions.sort_values(by=['Count'], ascending=False)
    completions.loc[:, ~completions.columns.isin(['Count', 'Percent'])] = completions.loc[:, ~completions.columns.isin(['Count', 'Percent'])].replace([0,1],['N','Y'])

    return completions

def completions_label_site(imaging, site, sites_info):
    # Get completions data for data subset
    if site == 'ALL':
        df = imaging.copy()
    elif site == 'MCC1':
        df = imaging[imaging['site'].isin(list(sites_info[sites_info['mcc']==1].site))].copy()
    elif site == 'MCC2':
        df = imaging[imaging['site'].isin(list(sites_info[sites_info['mcc']==2].site))].copy()
    else:
        df = imaging[imaging['site'] == site].copy()
    completions = get_completions(df)

    # Convert to multi-index
    multi_col = []
    for col in completions.columns[0:6]:
        t = ('Scan', col)
        multi_col.append(t)
    for col in completions.columns[6:]:
        t = (site, col)
        multi_col.append(t)
    completions.columns = multi_col

    return completions

def merge_completions(sites_list, imaging, sites_info):
    c = completions_label_site(imaging, sites_list[0], sites_info)
    for site in sites_list[1:]:
        c_site = completions_label_site(imaging, site, sites_info)
        c = c.merge(c_site, how='left', on=list(c.columns[0:6]))
    c = c.dropna(axis='columns', how='all').fillna(0)
    return c

# ----------------------------------------------------------------------------
# Heat matrix
# ----------------------------------------------------------------------------

def get_heat_matrix_df(qc, site, color_mapping_list):
    color_mapping_df = pd.DataFrame(color_mapping_list)
    color_mapping_df.columns= ['value','color']
    qc_cols = ['sub','ses', 'scan','rating']
    q = qc[(qc.site == site)][qc_cols]
    if len(q) >0:
        q['sub'] = q['sub'].astype(str)
        # Convert color designation to appropriate numeric value on the colorscale
        q2 = q.merge(color_mapping_df, how='left', left_on='rating', right_on='color')
        q3 = q2.sort_values(['sub','ses','scan']).drop_duplicates(['sub','ses','scan'],keep='last')
        q3['Scan'] = q3['ses'] + '-' + q3['scan']
        matrix_df = q3.pivot(index='sub', columns = 'Scan', values = 'value').fillna(0)

        # insert column to create grey border line in graph
        matrix_df.insert(5, "", [0.1] * len(matrix_df))
        # flatten table index
        matrix_df.columns.name = None
        matrix_df.index.name = None
    else:
        matrix_df = pd.DataFrame()
    return matrix_df

def get_stacked_bar_data(df, id_col, metric_col, cat_cols, count_col = None):
    if count_col:
        sb = df[cat_cols + [metric_col, id_col, count_col]].copy()
    else:
        count_col = 'count'
        sb = df[cat_cols + [metric_col, id_col]].copy()
        sb[count_col] = 1
    sb_grouped = sb[cat_cols+[metric_col, count_col]].groupby(cat_cols+[metric_col]).count()
    sb_grouped.reset_index(inplace=True)
    sb_grouped['Total N'] = sb_grouped.groupby(cat_cols)[count_col].transform('sum')
    sb_grouped['%'] = 100 * sb_grouped[count_col] / sb_grouped['Total N']
    return sb_grouped
