# ----------------------------------------------------------------------------
# PYTHON LIBRARIES
# ----------------------------------------------------------------------------
# Dash Framework
import dash_bootstrap_components as dbc
from dash import Dash, callback, clientside_callback, html, dcc, dash_table as dt, Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
import dash_daq as daq

import plotly.figure_factory as ff

# import local modules
from config_settings import *
from datastore_loading import *
from data_processing import *
from make_components import *
from styling import *

import logging


# Bar Chart options
bar_chart_options = {'None':'None', 'MCC':'mcc', 'Site':'site','Visit':'ses','Scan':'scan'}

# Set Version date
version_msg = 'Version Date: 04/22/24'
LOCAL_DATA_DATE = '04/22/24'

# Load local / asset data
sites_filepath = os.path.join(DATA_PATH,'sites.csv')
sites_info = pd.read_csv(sites_filepath)

# Get rescinded patients from local file until subjects API is working
rescinded_date = '07/11/2022'
rescinded_filepath = os.path.join(DATA_PATH,'rescinded.csv')
rescinded = pd.read_csv(rescinded_filepath)

# Load initial data from local store
initial_data_file_path = os.path.join(DATA_PATH,'initial_data.json')
with open(initial_data_file_path) as json_file:
    initial_data_dictionary = json.load(json_file)

# ----------------------------------------------------------------------------
# PROCESS DATA
# ----------------------------------------------------------------------------
# completions = get_completions(imaging)
# imaging_overview = roll_up(imaging)
mcc_dict = {'mcc': [1,1,1,2,2,2],
            'site':['UI','UC','NS','UM', 'WS','SH'],
            'site_facet': [1,2,3,1,2,3]
            }


scan_dict = {
    'DWI Received':'DWI',
    'T1 Received':'T1',
   '1st Resting State Received':'REST1',
   'fMRI Individualized Pressure Received':'CUFF1',
   'fMRI Standard Pressure Received':'CUFF2',
   '2nd Resting State Received':'REST2'}

icols = list(scan_dict.keys())
icols2 = list(scan_dict.values())

color_mapping_list = [(0.0, 'white'),(0.1, 'lightgrey'),(0.25, 'red'),(0.5, 'orange'),(0.75, 'yellow'),(1.0, 'green')]
# ---------------------------------
#   Data ETL
# ---------------------------------

release1_ids = list(pd.read_csv('assets/DataFreeze_1_ids.csv').subject_id)
release2_ids = list(pd.read_csv('assets/DataFreeze_2_ids.csv').subject_id)

def relative_date(nDays):
    today = datetime.today()
    relativeDate = (today - pd.Timedelta(days=nDays)).date()
    return relativeDate

def filter_imaging_by_date(imaging_df, date_col, start_date = None, end_date = None):
    '''Filter the imaging datatable using:
    start_date: select imaging records acquired on or after this date
    end_date: select imaging records acquired on or before this date'''
    filtered_imaging = imaging_df.copy()
    filtered_imaging[date_col] = pd.to_datetime(filtered_imaging[date_col]).dt.date
    
    if start_date and isinstance(start_date, date):
        filtered_imaging = filtered_imaging[filtered_imaging[date_col] >= start_date]

    if end_date and isinstance(end_date, date):
        filtered_imaging = filtered_imaging[filtered_imaging[date_col] <= end_date]

    return filtered_imaging
    
    
def filter_by_release(imaging, release_list):
    ''' Filter imaging list to only include the V1 visit for subjects from specific releases. '''
    filtered_imaging = imaging.copy()
    filtered_imaging = filtered_imaging[(filtered_imaging['subject_id'].isin(release_list)) & (filtered_imaging['visit']=='V1') ]
    
    return filtered_imaging

# ----------------------------------------------------------------------------
# APP Settings
# ----------------------------------------------------------------------------

external_stylesheets_list = [dbc.themes.SANDSTONE, 'https://codepen.io/chriddyp/pen/bWLwgP.css'] #  set any external stylesheets

app = Dash(__name__,
                external_stylesheets=external_stylesheets_list,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
                assets_folder=ASSETS_PATH,
                requests_pathname_prefix=REQUESTS_PATHNAME_PREFIX,
                suppress_callback_exceptions=True
                )
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger = logging.getLogger("imaging_ui")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(logging.INFO)


# ----------------------------------------------------------------------------
# DASH APP COMPONENT FUNCTIONS
# ----------------------------------------------------------------------------
def overview_heatmap(imaging):
    scan_dict = {'T1 Indicated':'T1',
       'DWI Indicated':'DWI',
       '1st Resting State Indicated':'REST1',
       'fMRI Individualized Pressure Indicated':'CUFF1',
       'fMRI Standard Pressure Indicated':'CUFF2',
       '2nd Resting State Indicated':'REST2',
                }

    icols = list(scan_dict.keys())
    icols2 = list(scan_dict.values())

    other_cols = ['site', 'subject_id', 'visit']
    i2 = pd.melt(imaging[other_cols + icols], id_vars=other_cols, value_vars = icols)
    hm = i2.groupby(['site','variable'])['value'].sum().reset_index()

    subjects = imaging[['subject_id','visit','site']].groupby(['site']).count().reset_index()
    subjects['Site'] = subjects['site'] + ' [' + subjects['visit'].astype('str') + ']'

    figdf = hm.merge(subjects, how='left', on='site')
    figdf['normed'] = 100 * figdf['value']/figdf['subject_id']
    figdf = figdf[['Site','variable','normed']].pivot(index='Site', columns='variable', values='normed')
    figdf.columns = ['REST1','REST2','DWI','T1','CUFF1','CUFF2']
    figdf = figdf[['T1','DWI','REST1','CUFF1','CUFF2','REST2']]
    figdf2 = figdf.applymap(lambda x: str(int(x)) + '%')

    heatmap_fig = ff.create_annotated_heatmap(figdf.to_numpy(),
                                  x=list(figdf.columns),
                                     y=list(figdf.index),
                                     annotation_text=figdf2.to_numpy(), colorscale='gray')

    return heatmap_fig

def create_image_overview(imaging_overview):
    overview_div = html.Div([
        dbc.Row([dbc.Col([
            html.H3('Overview')
        ])]),
        dbc.Row([
            dbc.Col([
                dt.DataTable(
                    id='tbl-overview', data=imaging_overview.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in imaging_overview.columns],
                    style_data_conditional=[
                        {
                            'if': {
                                'column_id': 'Total',
                            },
                            'backgroundColor': 'SteelBlue',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'row_index': imaging_overview.index[imaging_overview['site']=='All Sites'],
                            },
                            'backgroundColor': 'blue',
                            'color': 'white'
                        },
                    ]

                ),
            ],width=6),

        ]),

    ])
    return overview_div

def completions_div(completions_cols, completions_data, imaging):
    completions_div = [
        dbc.Row([dbc.Col([
            html.H3('Percent of imaged subjects completing a particular scan by site')
        ])]),
        dbc.Row([
            dbc.Col([
                # html.Div(overview_heatmap(imaging))
                dcc.Graph(id='graph-overview-heatmap', figure = overview_heatmap(imaging))
            ]),
        ]),
        dbc.Row([dbc.Col([
            html.H3('Overall completion of scans Y/N for each scan in acquisition order: T1, DWI, REST1, CUFF1, CUFF2, REST2)')
        ])]),
        dbc.Row([
            dbc.Col([
                dt.DataTable(
                    id='tbl',
                    columns = completions_cols,
                    data = completions_data,
                    filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    merge_duplicate_headers=True,
                ),
            ])
        ]),
    ]
    return completions_div


def build_boxplot(df):
    fig=go.Figure()

    data_length = (len(df['site'].unique()))
    df['Cuff1 Applied Pressure'] = df['Cuff1 Applied Pressure'].apply(pd.to_numeric, errors='coerce')

    for i, visit in enumerate(df['visit'].unique()):
        df_plot=df[df['visit']==visit]

        if len(df_plot) >0: 
            fig.add_trace(go.Box(x=df_plot['site'],
                                y=df_plot['Cuff1 Applied Pressure'],
                                meta = visit,
    #                              boxmean="sd",
                                line=dict(color=px.colors.qualitative.Plotly[i]),
                                boxpoints='suspectedoutliers',
                                name=visit,
                                offsetgroup=visit,
                                ))
        

        ## loop through the values you want to label and add them as annotations
    for i, visit in enumerate(df['visit'].unique()):
        for s in df['site'].unique():
            # Get settings for annotation
            xshift = 25
            if i == 0:
                xshift = -xshift
            else:
                xshift = xshift
            y_point = df[(df['site']==s) & (df['visit']==visit)]['Cuff1 Applied Pressure'].median()
            records = len(df[(df['site']==s) & (df['visit']==visit)]['Cuff1 Applied Pressure'])
            if records > 0:
                text = '<b>' + str(len(df[(df['site']==s) & (df['visit']==visit)]['Cuff1 Applied Pressure'])) + '</b>'

            else:
                text =''

            fig.add_annotation(
                    x=s,
                    y=y_point,
                    text=text,
                    showarrow=False,
                    xshift = xshift,
                    yshift=10,

            )

    fig.update_layout(boxmode='group',
                      xaxis_tickangle=0,
        autosize=False,
        width=1000,
        height=800,legend=dict(
                orientation="h"))

    return fig
# ----------------------------------------------------------------------------
# DASH APP LAYOUT FUNCTION
# ----------------------------------------------------------------------------
def load_tab_text():
    """ 
    Load content to display in the markdown component of the tabs from the 'tab_text' file in the github repo. 
    If this isn't accessible, load from a local file.
    """
    try:
        # try to load from github url
        tab_text_url = "https://raw.githubusercontent.com/TACC/a2cps-datastore-imaging/latest/src/assets/tab_text.json"
        resp = requests.get(tab_text_url)
        tab_text = json.loads(resp.text)
    except:
        # load from local json file if github url fails
        tab_text_path = "assets/tab_text.json"
        with open(tab_text_path) as f:
            tab_text = json.load(f)
    print(tab_text)
    return tab_text

def load_data_source(url_data_path, local_data_path, source):
    imaging, imaging_source = load_imaging(url_data_path, local_data_path, source)
    qc, qc_source = load_qc(url_data_path, local_data_path, source)

    return imaging, imaging_source, qc, qc_source

def get_processed_data(imaging, qc):
    completions = get_completions(imaging)
    imaging_overview = roll_up(imaging)
    indicated_received = get_indicated_received(imaging)
    print(indicated_received.columns)
    ratings = indicated_received.merge(qc, how='outer', left_on = ['Site','Subject','Visit','Scan'], right_on=['site','sub','ses','scan']).fillna('N/A')

    # stacked_bar_df = get_stacked_bar_data(qc, 'sub', 'rating', ['site','ses'])
    sites = list(imaging.site.unique())

    processed_data_dictionary = {
        'imaging': imaging.to_dict('records'),
        'qc': qc.to_dict('records'),
        'sites': sites,
        'completions': completions.to_dict('records'),
        'imaging_overview' : imaging_overview.to_dict('records'),
        'ratings' : ratings.to_dict('records'),
        'indicated_received' : indicated_received.to_dict('records'),
    }

    return processed_data_dictionary

def get_report_data_store(raw_data_store, selection = None, start_date: datetime = None, end_date = None):
    imaging = pd.DataFrame.from_dict(raw_data_store['imaging'])
    qc = pd.DataFrame.from_dict(raw_data_store['qc'])
    print(len(imaging))
    print(len(qc))
    print(selection)

    if imaging.empty or qc.empty:
        print('empty')
        completions = pd.DataFrame()
        imaging_overview =  pd.DataFrame()
        indicated_received =  pd.DataFrame()
        # stacked_bar_df =  pd.DataFrame()
        sites = []
    else:
        if not selection or selection =='all':
            imaging = imaging
            qc = qc 
        else:
            if selection == 'release 1':    
                imaging = filter_by_release(imaging, release1_ids)

            elif selection == 'release 2':    
                imaging = filter_by_release(imaging, release2_ids)

            else:    
                startDate = datetime.strptime(start_date, '%Y-%m-%d').date()
                endDate = datetime.strptime(end_date, '%Y-%m-%d').date()
                imaging = filter_imaging_by_date(imaging, 'acquisition_week', startDate, endDate)
            # Filter qc by the subject IDs in the image filter
            qc = filter_qc(qc, imaging)

    print(len(imaging))
    print(len(qc))

    processed_data_dictionary = get_processed_data(imaging, qc)

    return processed_data_dictionary

def load_imaging_api(api_url):
    api_address = api_url + 'imaging'
    app.logger.info('Requesting data from api {0}'.format(api_address))
    api_json = get_api_data(api_address)
    if 'error' in api_json:
        app.logger.info('Error response from datastore: {0}'.format(api_json))
        if 'error_code' in api_json:
            error_code = api_json['error_code']
            if error_code in ('MISSING_SESSION_ID', 'INVALID_TAPIS_TOKEN'):
                raise PortalAuthException

    if 'date' not in api_json or 'imaging' not in api_json['data'] or 'qc' not in api_json['data']:
        app.logger.info('Requesting data from api {0} to ignore cache.'.format(api_address))
        api_json = get_api_data(api_address, True)

    data_date = api_json['date']
    imaging = api_json['data']['imaging']
    qc = api_json['data']['qc']

    imaging_df = pd.DataFrame.from_dict(imaging)
    sites = list(imaging_df.site.unique())

    raw_data_dictionary = {
        'date': data_date,
        'imaging': imaging,
        'qc': qc,
        'imaging_source': 'api',
        'qc_source': 'api',
        'sites': sites,
    }

    return raw_data_dictionary

def local_data_as_dict(imaging, qc, LOCAL_DATA_DATE):
    sites = list(imaging.site.unique())
    raw_data_dictionary = {
        'date': LOCAL_DATA_DATE,
        'imaging': imaging.to_dict('records'),
        'qc': qc.to_dict('records'),
        'imaging_source': 'local',
        'qc_source': 'local',
        'sites': sites,
    }
    return raw_data_dictionary

def serve_raw_data_store(url_data_path, local_data_path, source):
    imaging, imaging_source, qc, qc_source = load_data_source(url_data_path, local_data_path, source)

    if imaging.empty or qc.empty:
        sites = []
    else:
        sites = list(imaging.site.unique())

    raw_data_dictionary = {
        'imaging': imaging.to_dict('records'),
        'qc': qc.to_dict('records'),
        'imaging_source': imaging_source,
        'qc_source': qc_source,
        'sites': sites,
    }
    return raw_data_dictionary

def create_data_stores(source, raw_data_dictionary):
    tab_text = load_tab_text()
    sites = raw_data_dictionary['sites']
    data_date = raw_data_dictionary['date']
    data_stores = html.Div([
        dcc.Store(id='session_data',  data = raw_data_dictionary), #storage_type='session',
        dcc.Store(id='tab_text', data=tab_text),
        dcc.Store(id='cache_data'),
        dcc.Store(id='report_data'),
        # html.P('Imaging Source: ' + data_dictionary['imaging_source']),
        # html.P('QC Source: ' + data_dictionary['qc_source']),
        create_content(source, data_date, sites)
    ])
    return data_stores

def create_content(source, data_date, sites):
    if source == 'api':
        source_msg = 'Data loaded from api: ' + data_date
    elif source == 'local':
        source_msg = 'Data loaded from local files dated ' + data_date
    else:
        source_msg = 'Data origin unknown'
    if len(sites) > 0:
        content = html.Div([
                    html.Div([
                        dbc.Row(id='content'),
                        dbc.Row([
                            dbc.Col([
                                html.H1('Imaging Overview Report' #, style={'textAlign': 'center'}
                                )
                            ], width=6),
                            dbc.Col([
                                html.Div([
                                    dcc.Dropdown(
                                        id='dropdown-date-range',
                                        options=[
                                            {'label': 'All records', 'value': 'all'},
                                            {'label': 'Custom Date Range', 'value': 'custom'},
                                            {'label': 'Data Release 1', 'value': 'release 1'},
                                            {'label': 'Data Release 2', 'value': 'release 2'},
                                            {'label': 'Recent (15 days)', 'value': '15'},
                                            {'label': '1 Month (30 days)', 'value': '30'},
                                            {'label': '6 Months (180 days)', 'value': '180'},
                                        ],
                                        value='all'
                                    ),
                                    html.Div(id='report-dates'),
                                    html.Div([
                                        dcc.DatePickerRange(
                                            id='date-picker-range',
                                            min_date_allowed=date(2021, 3, 29),
                                            start_date=date(2021, 3, 29),
                                            end_date = datetime.today().date(),
                                            # style={
                                            #     'display': 'none'
                                            # },
                                        ),  
                                    ]),
                                    html.Button('Re-load Report', id='btn-selections', n_clicks=0),
                                    html.Div(id='datediv'),
                            ]),
                            ], width=3)
                        ], justify='end', align='center'
                        ),
                        dbc.Row([
                            dbc.Col([
                                # html.P('Report Run date: ' + date.today().strftime('%B %d, %Y')),
                                html.P(source_msg),
                                html.P(version_msg),
                                ],
                            width=10),
                            dbc.Col([
                                # offcanvas
                                html.Div([
                                    html.P([' '], style={'background-color':'ForestGreen', 'height': '20px', 'width':'20px','float':'left'}),
                                    html.P(['no known issues'], style={'padding-left': '30px', 'margin': '0px'})
                                ]),
                                    html.Div([
                                        html.P([' '], style={'background-color':'Gold', 'height': '20px', 'width':'20px','float':'left', 'clear':'both'}),
                                        html.P(['minor variations/issues; correctable'], style={'padding-left': '30px', 'margin': '0px'})
                                    ]),
                                    html.Div([
                                        html.P([' '], style={'background-color':'FireBrick', 'height': '20px', 'width':'20px','float':'left', 'clear':'both'}),
                                        html.P(['significant variations/issues; not expected to be comparable'], style={'padding-left': '30px', 'margin': '0px'})
                                    ]),
                            ],width=2),
                        ], style={'border':'1px solid black'}),

                        dbc.Row([
                            dbc.Col([
                                dbc.Tabs(id="tabs", active_tab='tab-overview', children=[
                                    dbc.Tab(label='Overview', tab_id='tab-overview'),
                                    dbc.Tab(label='Completions', tab_id='tab-completions'),
                                    dbc.Tab(label='Pie Charts', tab_id='tab-pie'),
                                    dbc.Tab(label='Heat Map', tab_id='tab-heatmap'),
                                    dbc.Tab(label='Cuff Pressure', tab_id='tab-cuff'),
                                    dbc.Tab(label='Discrepancies', tab_id='tab-discrepancies'),
                                ]),
                            ],width=10),
                            dbc.Col([
                                dcc.Dropdown(
                                    id='dropdown-sites',
                                    options=[  #'UI', 'UC', 'NS', 'N/A', 'UM', 'WS', 'RU'
                                        {'label': 'All Sites', 'value': (',').join(sites)},
                                        {'label': 'MCC1', 'value': 'UI,UC,NS,RU'},
                                        {'label': 'MCC2', 'value': 'UM,WS,SH' },
                                        {'label': 'MCC1: University of Illinois at Chicago', 'value': 'UI' },
                                        {'label': 'MCC1: University of Chicago', 'value': 'UC' },
                                        {'label': 'MCC1: Endeavor Health', 'value': 'NS' },
                                        {'label': 'MCC1: Rush', 'value': 'RU' },
                                        {'label': 'MCC2: University of Michigan', 'value': 'UM' },
                                        {'label': 'MCC2: Wayne State University', 'value': 'WS' },
                                        {'label': 'MCC2: Corewell Health', 'value': 'SH' }
                                    ],
                                    # value = 'NS'
                                    multi=False,
                                    clearable=False,
                                    value=(',').join(sites)
                                ),
                            ], id='dropdown-sites-col',width=2),
                        ]),
                        dbc.Row([
                            dbc.Col([html.Div(id='tab-content')], className='delay')
                        ])

                    ]
                    , style={'border':'1px solid black', 'padding':'10px'}
                )
            ])
    else:
        content = html.Div([
            dbc.Alert("There has been a problem accessing the data API at this time. Please try again in a few minutes.", color="warning")
        ])
    return content

def serve_layout():
    try:
        # raw_data_dictionary = serve_raw_data_store(data_url_root, DATA_PATH, DATA_SOURCE)
        # try: #load data from api
        app.logger.info('serving layout using datastore: {0}'.format(DATASTORE_URL))
        raw_data_dictionary = load_imaging_api(DATASTORE_URL)
        source ='api'
        # except:
        #     imaging, qc = load_local_data(DATA_PATH)
        #     raw_data_dictionary = local_data_as_dict(imaging, qc, LOCAL_DATA_DATE)
        #     source = 'local'
        # try:
        page_layout =  html.Div([
        # change to 'url' before deploy
                # serve_data_stores('url'),
                create_data_stores(source, raw_data_dictionary),
                ], className='delay')

        # except:
        #     page_layout = html.Div(['There has been a problem accessing the data for this application.'])
        return page_layout
    except PortalAuthException:
        app.logger.warn('Auth error from datastore, asking user to authenticate')
        return html.Div([html.H4('Please login and authenticate on the portal to access the report.')])
    except Exception as ex:
        app.logger.warn('Exception serving layout {0}'.format(ex))
        return html.Div([html.H4('Error processing report data')])

app.layout = serve_layout


# ----------------------------------------------------------------------------
# DATA CALLBACKS
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Date Range / Filter type
# ----------------------------------------------------------------------------


@app.callback(
    Output("date-picker-range", "style"),
    Output("report-dates", "style"),
    Input("dropdown-date-range", "value"),
    State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date"),
    )
def update_visibility(customValue, startDate, endDate):    
    visible = {'display': 'block'}
    hidden = {'display': 'none'}
    if customValue == 'custom':
       return visible, hidden
    else:
        return hidden, visible

@app.callback(
    Output("date-picker-range", "start_date"),
    Output("date-picker-range", "end_date"),   
    Output("report-dates", "children"),
    Input("dropdown-date-range", "value"),
    )
def update_date_range(customValue):
    full_range_options = ['all','custom']
    start_date=date(2021, 3, 29)
    end_date = datetime.today().date()

    if customValue and customValue.isnumeric():
        start_date = relative_date(int(customValue))
    
    print(start_date, end_date)
    report_dates = 'Date Range: ' + start_date.strftime("%m/%d/%Y") + ' to ' + end_date.strftime("%m/%d/%Y")
    return start_date, end_date, report_dates

# @app.callback(
#     Output('cache_data', 'data'),
#     Output('datediv','children'),
#     Input('btn-selections','clicks'),
#     State('session_data', 'data'),
#     State("dropdown-date-range", "value"),
#     State("date-picker-range", "start_date"),
#     State("date-picker-range", "end_date"),
# )
# def filtered(clicks, raw_data, selection, startDate, endDate):
#     cache_data = {}
#     print(selection, startDate, endDate)
#     children = html.P("I'm here!")
#     return cache_data, children

# TO DO: Switch input to Re-load report button
@app.callback(
    Output('report_data', 'data'),
    Input('btn-selections','n_clicks'),
    State('session_data', 'data'),
    State("dropdown-date-range", "value"),
    State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date")
)
def filtered(clicks, rawData, selection, startDate, endDate):
    report_data = get_report_data_store(rawData, selection, startDate, endDate)
    print(startDate)
    print(endDate)
    return report_data

# Filter
@app.callback(
    Output('test-row','children'),
    Input('report_data', 'data')
)
def see_filtering(report_data):
    kids = html.Div()
    # kids = html.Div(json.dumps(report_data['qc']))
    return kids

@app.callback(Output("tab-content", "children"),
    Output('dropdown-sites-col','style'),
    Input("tabs", "active_tab"),
    State("tab_text","data")
    )    
def switch_tab(at, tab_text):
    if at == "tab-overview":
        overview = dcc.Loading(
                    id="loading-overview",
                    children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Markdown(tab_text['overview']['text'])
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H3('Scan sessions for pre-surgery (V1) and 3 month post surgery (V3) visits'),
                                    html.Div(id='overview_div')
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([html.H3('Quality ratings for individual scans (up to six scans per session: T1, DWI, REST1, CUFF1, CUFF2, REST2)')]),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id='graph_stackedbar_div')
                                    ], width=10),
                                dbc.Col([
                                    html.H3('Bar Chart Settings'),
                                    html.Label('Chart Type'),
                                    daq.ToggleSwitch(
                                            id='toggle_stackedbar',
                                            label=['Count','Stacked Percent'],
                                            value=False
                                        ),
                                    html.Label('Separate by Visit'),
                                    daq.ToggleSwitch(
                                            id='toggle_visit',
                                            label=['Combined','Split'],
                                            value=False
                                        ),

                                    html.Label('Chart Selection'),
                                    dcc.Dropdown(
                                        id='dropdown-bar',
                                    options=[
                                        {'label': ' Site and MCC', 'value': 1},
                                        {'label': ' Site', 'value': 2},
                                        {'label': ' MCC', 'value': 3},
                                        {'label': ' Combined', 'value': 4},
                                    ],
                                    multi=False,
                                    clearable=False,
                                    value=1
                                    ),
                                    ],width=2),
                                ]),
                        ])
                    ],
                    type="circle",
                )
        style = {'display': 'none'}
        return overview, style
    elif at == "tab-completions":
        completions = dcc.Loading(
                    id="loading-completions",
                    children=[
                        html.Div([
                            html.Div( dcc.Markdown(tab_text['completions']['text'])),
                            html.Div(id='completions_section')
                        ])
                    ],
                    type="circle",
                )
        return completions, {'display': 'block'}
    elif at == "tab-pie":
        pies = dcc.Loading(
                    id="loading-pie",
                    children=[
                        html.Div( dcc.Markdown(tab_text['pie']['text'])),
                        html.Div(id='pie_charts')
                        ],
                    type="circle",
                )
        return pies, {'display': 'block'}
    elif at == "tab-heatmap":
        heatmap = dcc.Loading(
                    id="loading-heatmap",
                    children=[
                        
                        html.Div( dcc.Markdown(tab_text['heatmap']['text'])),
                        html.Div([
                            dbc.Row([
                                dbc.Col([html.Div(id='heatmap')])
                            ]),
                        ])
                    ],
                type="circle",
            )
        return heatmap, {'display': 'block'}
    elif at == "tab-cuff":
        cuff = dcc.Loading(
                    id="loading-cuff",
                    children=[
                        
                        html.Div( dcc.Markdown(tab_text['cuff']['text'])),
                        html.Div([
                            dbc.Row([
                                dbc.Col([html.Div(id='cuff_section')])
                            ]),
                        ])
                    ],
                type="circle",
            )
        return cuff, {'display': 'block'}
    elif at == "tab-discrepancies":
        discrepancies = dcc.Loading(
                    id="loading-discrepancies",
                    children=[
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Markdown(tab_text['discrepancies']['text'])
                                ])
                            ]),

                            dbc.Row([
                                dbc.Col([html.Div(id='discrepancies_section')])
                            ]),
                        ])
                    ],
                    type="circle",
                )
        return discrepancies, {'display': 'block'}

    return html.P("This shouldn't ever be displayed...")
# Define callback to update graph_stackedbar

# Toggle Stacked bar toggle_stackedbar graph_stackedbar
@app.callback(
    Output('overview_div', 'children'),
    Input('report_data', 'data')
)
def update_overview_section(data):
    imaging_overview = pd.DataFrame.from_dict(data['imaging_overview'])
    return create_image_overview(imaging_overview)

@app.callback(
    Output('discrepancies_section', 'children'),
    Input('report_data', 'data')
)
def update_discrepancies_section(data):
    # Load imaging data from data store
    imaging = pd.DataFrame.from_dict(data['imaging'])
    df = pd.DataFrame.from_dict(data['indicated_received'])

    # Get records missing acquisition dates
    missing_dates_cols = ['site', 'subject_id', 'visit','dicom','bids', 'acquisition_week', 'Surgery Week'] 
    missing_dates = imaging[imaging.acquisition_week.isnull()][missing_dates_cols]

    # Rescinded patients in imaging
    cols = ['site', 'subject_id', 'visit',  'dicom',
    'bids', 'acquisition_week', 'Surgery Week']
    rescinded_imaging = imaging[imaging['subject_id'].isin(list(rescinded['main_record_id']))][cols]
    rescind_msg = 'Subjects who rescinded prior to ' + rescinded_date + ' but have records in the imaging file'

    # Get data for tables
    # df = get_indicated_received(imaging)
    df = pd.DataFrame.from_dict(data['indicated_received'])
    index_cols = ['Site','Subject','Visit']
    no_bids = df[df['BIDS']==0].sort_values(by=index_cols+['Scan'])
    mismatch = df[(df['DICOM']==1) & (df['Indicated'] != df['Received'])]

    missing_acquisition_table = dt.DataTable(
                id='tbl-no_acquisition', data=missing_dates.to_dict('records'),
                columns=[{"name": i, "id": i} for i in missing_dates.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                )
    
    no_bids_table = dt.DataTable(
                id='tbl-no_bids', data=no_bids.to_dict('records'),
                columns=[{"name": i, "id": i} for i in no_bids.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                )

    mismatch_table = dt.DataTable(
                id='tbl-mismatch', data=mismatch.to_dict('records'),
                columns=[{"name": i, "id": i} for i in mismatch.columns],
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                )


    discrepancies_div = html.Div([
        dbc.Col([
            html.H3("Data Missing Acquisition Week Field"),
            html.P("This information only avilable when 'All records' is selected.  These records are filtered out in other views." ),
            missing_acquisition_table
        ],width=6),
        dbc.Col([
            html.H3("BIDS value = 0"),
            no_bids_table
        ],width=6),
        dbc.Col([
            html.H3('Records with mismatch between indicated and received'),
            mismatch_table
        ],width=6),
    ])
    return discrepancies_div

@app.callback(
    Output('cuff_section', 'children'),
    Input('report_data', 'data')
)
def update_cuff_section(data):
    # Load imaging data from data store
    imaging = pd.DataFrame.from_dict(data['imaging'])
    fig = build_boxplot(imaging)
    cuff_div = html.Div([
             dbc.Col([
                 html.H3("Cuff1 Applied Pressure"),
                 dcc.Graph(id='boxplot_cuff1', figure=fig)
             ]),
     ])
    return cuff_div

@app.callback(
    Output('graph_stackedbar_div', 'children'),
    Input('report_data', 'data'),
    Input('toggle_stackedbar', 'value'),
    Input('toggle_visit', 'value'),
    Input('dropdown-bar', 'value'),
)
def update_stackedbar(report_data, type, visit, chart_selection):
    global mcc_dict
    # False = Count and True = Percent
    # return json.dumps(mcc_dict)
    if type:
        type = 'Percent'
    else:
        type = 'Count'

    qc = pd.DataFrame.from_dict(report_data['qc'])
    count_col='sub'
    color_col = 'rating'

    if chart_selection == 1:
        if visit:
            facet_row = 'ses'
        else:
            facet_row = None
        fig = bar_chart_dataframe(qc, mcc_dict, count_col, 'site', color_col, 'mcc', facet_row,  chart_type=type)
    else:
        if visit:
            x_col = 'ses'
        else:
            x_col = None
        if chart_selection == 2:
            fig = bar_chart_dataframe(qc, mcc_dict, count_col, x_col, color_col, 'site', chart_type=type)

        elif chart_selection == 3:
            fig = bar_chart_dataframe(qc, mcc_dict, count_col, x_col, color_col, 'mcc', chart_type=type)

        else:
            fig = bar_chart_dataframe(qc, mcc_dict, count_col, x_col, color_col, chart_type=type)

    return [html.P(visit), dcc.Graph(id='graph_stackedbar', figure=fig)]

@app.callback(
    Output('completions_section', 'children'),
    Input('dropdown-sites', 'value'),
    State('report_data', 'data')
)
def update_image_report(sites, data):
    imaging = pd.DataFrame.from_dict(data['imaging'])
    sites_list = ['ALL', 'MCC1', 'MCC2'] + list(imaging.site.unique())
    completions = merge_completions(sites_list, imaging, sites_info)

    # Conver tuples to multiindex then prepare data for dash data table
    completions.columns = pd.MultiIndex.from_tuples(completions.columns)
    completions_cols, completions_data = datatable_settings_multiindex(completions)
    ct = completions_div(completions_cols, completions_data, imaging)
    # kids = [html.P(c) for c in list(completions.columns)]
    # return kids
    return ct

@app.callback(
    Output('pie_charts', 'children'),
    Input('dropdown-sites', 'value'),
    Input('report_data', 'data'),
    State('dropdown-sites', 'options')
)
def update_pie(sites, report_data, options):
    sites_list = sites.split(",")
    ratings = pd.DataFrame.from_dict(report_data['ratings'])
    site_label = [x['label'] for x in options if x['value'] == sites]
    pie_df = ratings[ratings['Site'].isin(sites_list)]

    pie_charts = [
        dbc.Row([dbc.Col([
            html.H3('Percent of returns by Scan type'),
        ])]),
        dbc.Row([
            html.H4(site_label),
            dcc.Graph(id='pie_main', figure=make_pie_chart(pie_df))
        ])
    ]

    if len(sites_list) > 1:
        fig_height = len(sites_list) * 300
        fig=make_pie_chart(pie_df, facet_row='Site')
        fig.update_layout(
            autosize=False,
            width =1500,
            height=fig_height)

        add_div =dbc.Row([
            html.H4('Breakout by Site'),
            dcc.Graph(id='pie_main', figure=fig)
        ])
        pie_charts.append(add_div)

    return pie_charts



@app.callback(
    Output('heatmap', 'children'),
    Input('dropdown-sites', 'value'),
    Input('report_data', 'data')
)
def update_heatmap(sites, data):
    global color_mapping_list
    sites_list = sites.split(",")
    qc = pd.DataFrame.from_dict(data['qc'])

    if len(sites_list) == 1:
        df  = get_heat_matrix_df(qc, sites, color_mapping_list)
        if not df.empty:
            fig_heatmap = generate_heat_matrix(df.T, color_mapping_list) # transpose df to generate horizontal graph
            heatmap = html.Div([
                dcc.Graph(id='graph_heatmap', figure=fig_heatmap)
            ])
        else:
            heatmap = html.Div([
                html.H4('There is not yet data for this site')
            ], style={'padding': '50px', 'font-style': 'italic'})
    else:
        heatmap = html.Div([
            html.H4('Please select a single site from the dropdown above to see a Heatmap of Image Quality')
        ], style={'padding': '50px', 'font-style': 'italic'})    # f = generate_heat_matrix(get_heat_matrix_df(qc, sites), colors)
    # heatmap = dcc.Graph(figure=f)

    return heatmap


# ----------------------------------------------------------------------------
# RUN APPLICATION
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True,)
else:
    server = app.server
