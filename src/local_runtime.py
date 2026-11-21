"""Local model transport and readiness; never probes or falls back to cloud AI."""
import json
import urllib.error
import urllib.parse
import urllib.request

OLLAMA = 'http://127.0.0.1:11434'


class LocalServiceUnavailable(RuntimeError):
    """Transient local model failure, distinct from source/content failures."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError('Local model redirects are disabled')


def open_local(request, timeout):
    url = request if isinstance(request, str) else request.full_url
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme, parsed.hostname, parsed.port) != ('http', '127.0.0.1', 11434) or parsed.username or parsed.password:
        raise ValueError('Model endpoint must be the local Ollama service')
    # Ignore system proxies, including when NO_PROXY is not configured.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(request, timeout=timeout)


def readiness(qa_model):
    required = {'qa': qa_model, 'import_text': 'qwen3:8b', 'import_vision': 'qwen3-vl:4b-instruct'}
    try:
        with open_local(OLLAMA + '/api/tags', timeout=2) as response:
            body = json.load(response)
        models = body['models']
        if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
            raise ValueError('Invalid model inventory')
        installed = {item.get('name') for item in models if isinstance(item.get('name'), str)}
    except (OSError, ValueError, KeyError, TypeError):
        return {'status': 'unavailable', 'cloud_fallback': False,
                'reason': '本機模型服務無法連線或回應無效；請啟動 Ollama 後重試。'}
    capabilities = {role: {'model': model, 'installed': model in installed} for role, model in required.items()}
    return {'status': 'ready' if all(c['installed'] for c in capabilities.values()) else 'models_missing',
            'cloud_fallback': False, 'capabilities': capabilities,
            'inference_verified': False}
