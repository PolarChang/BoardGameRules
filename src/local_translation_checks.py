"""Conservative translation diagnostics, not semantic correctness certification."""
import re


def translation_warnings(original, translation):
    warnings = []
    source = re.sub(r'\s', '', original)
    translated = re.sub(r'\s', '', translation)
    if len(re.findall(r'[A-Za-z]', original)) >= 400 and len(translated) < len(source) * .18:
        warnings.append('譯文相對原文明顯偏短，可能遺漏段落／答案，須逐段核對。')
    if re.search(r'\bround(?:ed|ing)?[\s-]+up\b', original, re.I) and '四捨五入' in translation:
        warnings.append('原文包含無條件進位，但譯文出現四捨五入，須核對對應句。')
    return warnings
