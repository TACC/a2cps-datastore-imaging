import os
import json
import flask
import requests
import logging

# ---------------------------------
#   MOVE THIS TO REFERENCE FROM ENV
# ---------------------------------
DATASTORE_URL = os.environ.get("DATASTORE_URL","url not found")
DATASTORE_URL = os.path.join(DATASTORE_URL, "api/")
logger  = logging.getLogger("imaging_app")

# ---------------------------------
#   Get Data From datastore
# ---------------------------------

class PortalAuthException(Exception):
    '''Custom Exception for issues with Authentication'''

def get_api_data(api_address, ignore_cache=False):
    api_json = {}
    try:
        params = {}
        if ignore_cache:
            params = {'ignore_cache':True}
        response = requests.get(api_address, params=params, cookies=flask.request.cookies)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warn(e)
        api_json['json'] = 'error: {}'.format(e)
        return api_json
