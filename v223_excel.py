"""Compact Excel research workbook for V22.3 reports."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def build_excel(run: Path, result: dict):
    run = Path(run)
    out = run / 'V22_TEST_REPORT.xlsx'
    try:
        import openpyxl  # noqa: F401
    except Exception:
        return None
    csvs = [
        ('候選排行','candidate_ranking.csv'),('階段績效','strategy_by_stage.csv'),
        ('每月決策','monthly_decisions.csv'),('因子敏感度','factor_sensitivity.csv'),
        ('走期因子IC','factor_ic_summary.csv'),('自適應權重','adaptive_weight_path.csv'),
        ('權重建議','static_weight_suggestions.csv'),('市場階段','market_stage.csv'),
        ('階段轉換','regime_transitions.csv'),('失敗分析','failure_analysis.csv'),
        ('資料品質','data_quality.csv'),('最終確認','final_confirmation_records.csv'),
    ]
    summary=[]
    for h,m in result.get('models',{}).items():
        summary.append({'週期':h,'母池':m.get('pool'),'策略':m.get('choice',{}).get('name'),
            '模型類型':m.get('choice',{}).get('model_type','manual'),'開發分數':m.get('choice_score'),
            '判定':m.get('verdict'),'最終保留期已查看':m.get('final_tail_inspected')})
    with pd.ExcelWriter(out,engine='openpyxl') as xw:
        pd.DataFrame(summary).to_excel(xw,sheet_name='摘要',index=False)
        pd.DataFrame([{'項目':'version','值':result.get('version')},
                      {'項目':'mode','值':result.get('mode')},
                      {'項目':'pool_rule','值':result.get('pool_rule')},
                      {'項目':'data_scope','值':json.dumps(result.get('data_scope',{}),ensure_ascii=False)},
                      {'項目':'automatic_orders_authorized','值':result.get('automatic_orders_authorized')},
                      {'項目':'global_optimum_proven','值':result.get('global_optimum_proven')}]).to_excel(xw,sheet_name='研究設定',index=False)
        for sheet,fn in csvs:
            p=run/fn
            if p.is_file():
                try:pd.read_csv(p).to_excel(xw,sheet_name=sheet[:31],index=False)
                except Exception:pass
        for sheet,fn in [('Profiles','profiles.json'),('Settings','settings.json')]:
            p=run/fn
            if p.is_file():
                obj=json.loads(p.read_text(encoding='utf-8'))
                pd.DataFrame({'JSON':[json.dumps(obj,ensure_ascii=False,indent=2)]}).to_excel(xw,sheet_name=sheet,index=False)
    return out
