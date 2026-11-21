"""Allowlisted, non-secret local settings carried with a restored installation."""
import json
import re

DEFAULTS = {
    'LOCAL_QA_MODEL':'qwen3:8b', 'LOCAL_CATEGORY_MODEL':'qwen3:8b',
    'LOCAL_M3_QA_ENABLED':'false', 'LOCAL_QA_REASONING_ENABLED':'false',
    'LOCAL_QA_DAILY_LIMIT':'100', 'LOCAL_CATEGORY_DAILY_LIMIT':'256',
    'LOCAL_QA_RESERVATION_SECONDS':'900',
}
for family,day,document in (
    ('IMPORT',(256,2000000,7200),(512,4000000,14400)),
    ('QA',(100,500000,3600),(300,1500000,10800)),
    ('CATEGORY',(256,1000000,7200),(256,1000000,7200)),
    ('CPU',(1000,2000000,7200),(2000,4000000,14400)),
):
    for scope,values in (('DAY',day),('DOCUMENT',document),('WORK',(8,100000,1920))):
        for metric,value in zip(('CALLS','TOKENS','SECONDS'),values):
            DEFAULTS[f'LOCAL_{family}_{scope}_{metric}']=str(value)


def validate(values):
    if not isinstance(values,dict) or set(values)!=set(DEFAULTS):
        raise ValueError('Invalid local runtime settings scope')
    for name,value in values.items():
        if not isinstance(value,str): raise ValueError('Invalid local runtime setting')
        pattern = r'[A-Za-z0-9_./:-]{1,120}' if name.endswith('_MODEL') else r'true|false' if name.endswith('_ENABLED') else r'[0-9]{1,12}'
        if not re.fullmatch(pattern,value): raise ValueError('Invalid local runtime setting: '+name)
    return values


def capture():
    from src.gemini_usage import setting
    return validate({name:setting(name,default) for name,default in DEFAULTS.items()})


def read(path):
    return validate(json.loads(path.read_text(encoding='utf-8')))
