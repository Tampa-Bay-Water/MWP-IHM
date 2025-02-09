import os
import sys
import yaml

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CONFIG_FILE = os.path.join(ROOT_DIR,'config.yaml')
with open(CONFIG_FILE, 'r') as file:
    try:
        CONFIG = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"\033[91mError parsing YAML file: {e}\033[0m", file=sys.stderr)
        CONFIG = None

IS_DEBUGGING = CONFIG['general']['IS_DEBUGGING']
MAX_NUM_PROCESSES = CONFIG['general']['MAX_NUM_PROCESSES']
INTB_VERSION = CONFIG['general']['INTB_VERSION']
FILE_REGRESSION_PARAMS = CONFIG['general']['FILE_REGRESSION_PARAMS']

FIG_TITLE_FONTSIZE = CONFIG['plotting']['FIG_TITLE_FONTSIZE']
TITLE_FONTSIZE = CONFIG['plotting']['TITLE_FONTSIZE']
AX_LABEL_FONTSIZE = CONFIG['plotting']['AX_LABEL_FONTSIZE']

if INTB_VERSION==1:
    # Period of Analysis INTB1
    POA = [CONFIG['INTB1']['POA_sdate'],CONFIG['INTB1']['POA_edate']]
    # Calibration period INTB1
    CAL_PERIOD = [CONFIG['INTB1']['CAL_PERIOD_sdate'],CONFIG['INTB1']['CAL_PERIOD_edate']]
else:
    # Period of Analysis INTB1
    POA = [CONFIG['INTB2']['POA_sdate'],CONFIG['INTB2']['POA_edate']]
    # Calibration period INTB1
    CAL_PERIOD = [CONFIG['INTB2']['CAL_PERIOD_sdate'],CONFIG['INTB2']['CAL_PERIOD_edate']]

print(f'{IS_DEBUGGING}: {type(IS_DEBUGGING)}')
print(f'{INTB_VERSION}: {type(INTB_VERSION)}')
print(f'{FILE_REGRESSION_PARAMS}: {type(FILE_REGRESSION_PARAMS)}')
print(f'{POA}: {type(POA)}')