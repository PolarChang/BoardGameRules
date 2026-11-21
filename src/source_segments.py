"""Lossless bounded source segmentation; boundaries never imply separate rules."""
import re


def split_source(text, limit=2800):
    """Prefer sentence/line boundaries, preserving every character and offset."""
    if limit < 1:
        raise ValueError('limit must be positive')
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        forced = False
        if end < len(text):
            candidates = [m.end() for m in re.finditer(r'[。！？](?:[」』”])?|[.!?](?=\s)', text[start:end])]
            if candidates:
                end = start + candidates[-1]
            else:
                lines = list(re.finditer(r'\n', text[start:end]))
                spaces = lines or list(re.finditer(r'\s+', text[start:end]))
                if spaces:
                    end = start + spaces[-1].end()
                forced = True
        yield {'text': text[start:end], 'start': start, 'end': end,
               'continues_before': start > 0, 'continues_after': end < len(text),
               'forced_boundary': forced}
        start = end


def locate_text(page, text):
    """Return bounding blocks only when they account for all normalized text.

    A block rectangle is not a word rectangle. Explicit granularity prevents
    old/page-only locations from being presented as exact source coordinates.
    """
    compact = lambda value: re.sub(r'\s+', '', re.sub(r'(?<=\w)-\n(?=\w)', '', value))
    target = compact(text)
    remaining = target
    boxes = []
    for block in page.get_text('blocks'):
        if block[6] != 0:
            continue
        value = compact(block[4])
        if not value:
            continue
        if target and target in value:
            boxes = [list(block[:4])]
            remaining = ''
            break
        if value in remaining:
            boxes.append(list(block[:4]))
            remaining = remaining.replace(value, '', 1)
    if remaining or not boxes:
        return {'bbox': list(page.rect), 'granularity': 'page_only'}
    return {'bbox': [min(b[0] for b in boxes), min(b[1] for b in boxes),
                     max(b[2] for b in boxes), max(b[3] for b in boxes)],
            'granularity': 'text_blocks', 'regions': boxes}
