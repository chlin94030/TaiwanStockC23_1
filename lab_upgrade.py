"""Explicit version migration: reuse price cache/end date, never old model settings.
A completed run of this version is not restarted merely because today changed.
"""
from dataclasses import replace
from lab_config import Config, VERSION


def run_config(default,latest,protocol):
    previous=latest.get('config') or {}
    if not previous or previous.get('source_mode')!=default.source_mode:return default
    if protocol.get('version')==VERSION:
        return Config(**previous).resolved()
    if default.freeze_existing_data_end and previous.get('end'):
        # Keep the exact observed era. New factor code/settings remain explicit.
        return replace(default,start=previous.get('start',default.start),end=previous['end']).resolved()
    return default


def reuse_collection_context(latest,run,cfg):
    """Reuse only the old frozen roster/calendar for an identical data window.
    Do not reuse features, labels, models or performance. No raw cache is erased.
    """
    from pathlib import Path
    import re
    from lab_io import read_json,atomic_json,sha
    previous=latest.get('config') or {}
    if cfg.source_mode!='auto_current_roster' or previous.get('source_mode')!=cfg.source_mode:
        return {}
    if any(previous.get(k)!=getattr(cfg,k) for k in ('start','end')) or not latest.get('run'):
        return {}
    old=Path(latest['run']);run=Path(run)
    if old.resolve()==run.resolve():return {}
    roster=read_json(old/'roster.json',[]);calendar=read_json(old/'calendar.json',[])
    if not roster or not calendar:return {}
    if not all(isinstance(r,dict) and re.fullmatch(r'[1-9][0-9]{3}\.(TW|TWO)',str(r.get('ticker',''))) for r in roster):
        return {}
    if len({r['ticker'] for r in roster})!=len(roster):return {}
    if not all(isinstance(d,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',d) and cfg.start<=d<=cfg.end for d in calendar):
        return {}
    if calendar!=sorted(set(calendar)):return {}
    if not (run/'roster.json').exists():atomic_json(run/'roster.json',roster)
    if not (run/'calendar.json').exists():atomic_json(run/'calendar.json',calendar)
    info={'from_run':old.name,'same_requested_start':cfg.start,'same_requested_end':cfg.end,
          'roster_sha':sha(old/'roster.json'),'calendar_sha':sha(old/'calendar.json'),
          'reused_features_or_outcomes':False,'reason':'SAME_FROZEN_PRICE_WINDOW_NOT_NEW_HOLDOUT'}
    atomic_json(run/'collection_reuse.json',info)
    return info
