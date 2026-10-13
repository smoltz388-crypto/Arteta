# ==============================================================================
# 自動同步之 FinLab 小市值高成長量化交易策略
# ==============================================================================
import os
import pandas as pd
import requests
from finlab import data
from finlab.backtest import sim
from google.colab import userdata

# ==============================================================================
# Telegram 發送設定 (安全讀取金鑰)
# ==============================================================================
try:
    TELEGRAM_BOT_TOKEN = userdata.get('TELEGRAM_BOT_TOKEN')
except Exception:
    TELEGRAM_BOT_TOKEN = None

try:
    TELEGRAM_CHAT_ID = userdata.get('TELEGRAM_CHAT_ID')
except Exception:
    TELEGRAM_CHAT_ID = None

def send_telegram_message(token, chat_id, text):
    """發送 HTML 格式訊息至 Telegram Bot"""
    if not token or not chat_id:
        print('⚠️ 尚未設定有效的 Telegram Bot Token 或 Chat ID，跳過 Telegram 推播。')
        return None

    # 如果使用者金鑰前後被夾帶了雙引號，在此做一層清洗
    clean_token = str(token).strip('"').strip("'")
    clean_chat_id = str(chat_id).strip('"').strip("'")

    url = f'https://api.telegram.org/bot{clean_token}/sendMessage'
    payload = {'chat_id': clean_chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        response = requests.post(url, data=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f'❌ Telegram 發送失敗: {e}')
        return None

def main():
    print('正在讀取 FinLab 資料庫與計算策略因子...')
    mv = data.get('etl:market_value')
    rev = data.get('monthly_revenue:當月營收')
    close = data.get('price:收盤價')
    ind_df = data.get('security_industry_themes')

    # 營收年增率 (YoY)
    rev_growth = (rev / rev.shift(12)) - 1

    # 產業分類處理
    stock_to_category = ind_df.set_index('stock_id')['category'].to_dict()
    categories = pd.Series(stock_to_category).reindex(close.columns)

    # 計算個股超額報酬 (相對產業)
    ret20 = close.pct_change(20, fill_method=None)
    ind_ret20 = ret20.T.groupby(categories).transform('mean').T
    excess_ret = ret20 - ind_ret20

    # 營收年增率對齊日線時間軸
    rev_growth_aligned = rev_growth.reindex(close.index, method='ffill')

    # 【市值篩選設定】預設篩選市值最小的前 100 檔
    market_value_threshold = 100
    size_filter = mv.rank(axis=1, ascending=True) <= market_value_threshold

    # 綜合排名
    rank_sum = rev_growth_aligned.rank(axis=1, pct=True) + excess_ret.rank(axis=1, pct=True)

    # 套用市值過濾並選出前 20 檔
    filtered_rank = rank_sum[size_filter]
    position = filtered_rank.is_largest(20)

    # 執行模擬
    report = sim(position, resample='M', upload=False)

    ar = report.metrics.annual_return() * 100
    sr = report.metrics.sharpe_ratio()
    md = report.metrics.max_drawdown() * 100
    print(f'\n策略歷史回測績效: 年化報酬率(AR)={ar:.2f}% | 夏普值(SR)={sr:.2f} | 最大回撤(MD)={md:.2f}%\n')

    # 輸出持股與推播
    pos_info = report.position_info()
    df = pd.DataFrame(pos_info).T
    df = df[df.index.str.contains('^[0-9]')]

    hold_stocks = df[df['status'] == 'hold'].copy()
    enter_stocks = df[df['status'] == 'enter'].copy()
    latest_date = df['entry_date'].max()[:10]

    # 訊息組合
    msg = f'<b>📊 【FinLab 小市值高成長 - 換股通知】</b>\n'
    msg += f'📅 結算/換股日期：<code>{latest_date}</code>\n'
    msg += '------------------------------------\n\n'

    msg += f'<b>🟢 【即將進場】({len(enter_stocks)} 檔)</b>\n'
    if not enter_stocks.empty:
        for stock, row in enter_stocks.iterrows():
            price = f'${row["entry_price"]:.2f}' if pd.notnull(row['entry_price']) else '-'
            msg += f'• <b>{stock}</b> | 參考價: {price}\n'
    else:
        msg += '• 無新進場股票\n'
    msg += '\n'

    msg += f'<b>🔵 【繼續持有】({len(hold_stocks)} 檔)</b>\n'
    if not hold_stocks.empty:
        for stock, row in hold_stocks.iterrows():
            ret = row['return'] * 100 if pd.notnull(row['return']) else 0
            ret_str = f'+{ret:.2f}%' if ret >= 0 else f'{ret:.2f}%'
            msg += f'• <b>{stock}</b> | 報酬: <code>{ret_str}</code>\n'
    else:
        msg += '• 無持有股票\n'

    msg += '\n------------------------------------\n'
    msg += '🤖 <i>訊息由 FinLab 量化系統自動發送</i>'

    res = send_telegram_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
    if res and res.get('ok'):
        print('✅ Telegram 持股推播發送成功！')

if __name__ == "__main__":
    main()
