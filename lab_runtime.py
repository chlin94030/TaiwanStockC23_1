"""Pure progress / duration diagnostics. No data collection or Streamlit dependency."""
from datetime import datetime, timezone

FREE_PER_HOUR = 260
TOKEN_FALLBACK_PER_HOUR = 520


def rate(token_present=False, safe_limit=None):
    if safe_limit:
        try:return max(60,int(safe_limit))
        except Exception:pass
    return TOKEN_FALLBACK_PER_HOUR if token_present else FREE_PER_HOUR


def estimated_hours(requests_remaining, token_present=False, safe_limit=None):
    return max(0, int(requests_remaining)) / rate(token_present,safe_limit)


def hours_message(total, completed=0, token_present=False, safe_limit=None, provider='FINMIND'):
    remaining=max(0,int(total)-int(completed))
    if provider=='FINLAB_BULK':
        return ('FinLab bulk 模式不逐檔送出約2,000個歷史請求；先整表取得全市場成交金額，'
                '再只對歷史母池聯集建立完整價格/因子。實際時間取決於帳號權限、下載大小與Cloud記憶體。')
    if provider=='YAHOO_FAST':
        return ('Yahoo批次快速建庫：不受FinMind 260次/小時節奏限制；會分批取得全市場OHLCV，'
                '母池流動性使用((高+低+收)/3)×成交量代理。這是快速開發資料，不冒充交易所實際成交金額。')
    if provider=='OFFICIAL_BULK':
        return ('官方整批精確模式：每個交易日只向TWSE與TPEx各取一份全市場行情，'
                '不是每檔股票各送一次請求；使用交易所實際成交金額。首次歷史回補仍需一段時間，但通常遠短於7.6小時。')
    r=rate(token_present,safe_limit);hours=estimated_hours(remaining,token_present,safe_limit)
    if remaining==0:return '逐檔歷史量值收集已到目前階段的最後一檔。'
    return (f'如剩餘 {remaining:,} 檔均需向 FinMind 請求，目前安全節奏約 {r:,} 次/小時，'
            f'純請求節奏約 {hours:.1f} 小時；實際配額會由帳號API動態讀取，已有快取會更快。')


def seconds_since(iso_value, now=None):
    if not iso_value:return None
    try:
        parsed=datetime.fromisoformat(str(iso_value))
        if parsed.tzinfo is None:return None
        delta=(now or datetime.now(timezone.utc))-parsed
        return max(0,int(delta.total_seconds()))
    except (ValueError,TypeError,OverflowError):return None


def progress_caption(state, token_present=False):
    phase=state.get('phase')
    if phase=='BULK_AMOUNT':return 'FinLab整表建立母池中：不是逐檔FinMind請求。完成後只處理固定母池聯集。'
    if phase in ('YAHOO_FAST','YAHOO_SELECTED','YAHOO_ADJUSTED','YAHOO_REPAIR'):return 'Yahoo快速建庫／完整度修復中：不是FinMind逐檔260次/小時模式；只重抓稀疏ticker，已完成批次會保留。'
    if phase=='OFFICIAL_FALLBACK':return 'Yahoo歷史完整度不足，正切換TWSE/TPEx官方整市場回補；不會退回7小時FinMind逐檔模式。'
    if phase in ('OFFICIAL_BULK','OFFICIAL_SELECTED'):return 'TWSE/TPEx官方整批回補中：每個日期各市場只取一份全市場資料；已完成日期會保留快取。'
    if phase!='AMOUNT':return ''
    total=state.get('total');done=state.get('done',0)
    if not isinstance(total,int) or not isinstance(done,int) or total<=0:return ''
    return hours_message(total,done,token_present,state.get('finmind_safe_limit'))
