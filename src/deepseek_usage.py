"""Read-only account status. Never sends inference requests or exposes credentials."""
import json
import os
import urllib.request
from pathlib import Path

from dotenv import dotenv_values


def account_status():
    local = dotenv_values(Path(__file__).resolve().parent.parent / '.env')
    key = os.environ.get('DEEPSEEK_API_KEY', local.get('DEEPSEEK_API_KEY', ''))
    result = {'configured': bool(key), 'free_only': True, 'balances': [],
              'token_usage': None, 'usage_note': '尚未開始記錄模型請求；帳戶歷史 Token 用量請至 DeepSeek 平台查看。'}
    if not key:
        return result
    request = urllib.request.Request('https://api.deepseek.com/user/balance',
                                     headers={'Authorization': f'Bearer {key}'})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
        result['balances'] = [{name: item.get(name, '') for name in
                               ('currency', 'granted_balance', 'topped_up_balance', 'total_balance')}
                              for item in payload['balance_infos']]
        result['available'] = bool(payload.get('is_available'))
    except Exception:
        # Provider bodies and exception messages can contain credentials.
        raise RuntimeError('無法讀取 DeepSeek 餘額，請確認金鑰與網路連線。') from None
    return result
