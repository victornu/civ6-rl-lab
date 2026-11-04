"""Lossless, bounded byte inputs for rendered diplomatic text only.

Native identifiers and reasons have already been removed by the public
projection. Byte 0 is padding; UTF-8 bytes are shifted by one. No vocabulary
fitting, normalization or truncation is performed here.
"""
MAX_TEXT_BYTES = 4096
MAX_TOTAL_BYTES = 131072
MAX_MENU_ROWS = 64
TEXT_LAYOUT = {
    'schema': 'civ6rl.public_dialogue_bytes987.v1',
    'encoding': 'utf8_byte_plus_one_zero_padding',
    'context': ['speaker_leader', 'statement_text', 'reason_text'],
    'menu': ['text', 'tooltip'],
    'menu_flags': ['disabled', 'tooltip_present', 'public_ordinal_div_64'],
    'max_text_bytes': MAX_TEXT_BYTES, 'max_total_bytes': MAX_TOTAL_BYTES,
    'max_menu_rows': MAX_MENU_ROWS,
}


def dialogue_bytes(value, candidate_count, *, candidate_mask=None):
    """Validate a closed public shape before creating any neural input."""
    if type(candidate_count) is not int or candidate_count < 0:
        raise ValueError('invalid candidate count')
    if value is None:
        if candidate_mask is not None:
            if (type(candidate_mask) is not list or len(candidate_mask) != candidate_count
                    or any(type(mask) is not bool for mask in candidate_mask)
                    or any(candidate_mask)):
                raise ValueError('active candidates require public dialogue text')
        return {'context': [[], [], []], 'menu': [], 'flags': [],
                'candidate_rows': [-1] * candidate_count, 'present': False}
    if type(value) is not dict or set(value) != {'context', 'menu', 'candidate_menu_indices'}:
        raise ValueError('public dialogue fields differ')
    if type(value['context']) is not list or len(value['context']) != 3:
        raise ValueError('public dialogue context differs')
    menu = value['menu']
    if type(menu) is not list or not 1 <= len(menu) <= MAX_MENU_ROWS:
        raise ValueError('public dialogue menu bound')
    total = 0

    def encode(text):
        nonlocal total
        if type(text) is not str:
            raise ValueError('public dialogue text must be a string')
        data = text.encode('utf8')
        if len(data) > MAX_TEXT_BYTES:
            raise ValueError('public dialogue text byte bound')
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError('public dialogue total byte bound')
        return [byte + 1 for byte in data]

    context = [encode(text) for text in value['context']]
    rows, flags, enabled = [], [], []
    for index, row in enumerate(menu):
        if type(row) is not dict or set(row) != {'text', 'tooltip', 'disabled'} or type(row['disabled']) is not bool:
            raise ValueError('public dialogue menu fields differ')
        tooltip = row['tooltip']
        rows.append([encode(row['text']), encode('' if tooltip is None else tooltip)])
        flags.append([float(row['disabled']), float(tooltip is not None), index / MAX_MENU_ROWS])
        if not row['disabled']:
            enabled.append(index)
    indices = value['candidate_menu_indices']
    if type(indices) is not list or len(indices) != candidate_count or any(type(index) is not int for index in indices):
        raise ValueError('public dialogue candidates differ from complete enabled menu')
    if candidate_mask is None:
        if candidate_count > 0 and indices != enabled:
            raise ValueError('public dialogue candidates differ from complete enabled menu')
    else:
        if (type(candidate_mask) is not list or len(candidate_mask) != candidate_count
                or any(type(mask) is not bool for mask in candidate_mask)):
            raise ValueError('dialogue candidate mask differs from candidate rows')
        active_indices = [index for index, mask in zip(indices, candidate_mask) if mask]
        masked_indices = [index for index, mask in zip(indices, candidate_mask) if not mask]
        if (any(index != -1 for index in masked_indices)
                or any(index < 0 or index >= len(menu) for index in active_indices)
                or len(active_indices) != len(set(active_indices))
                or (candidate_count > 0 and enabled and not active_indices)
                or (active_indices and set(active_indices) != set(enabled))):
            raise ValueError('active dialogue candidates lack a unique enabled menu row')
    return {'context': context, 'menu': rows, 'flags': flags,
            'candidate_rows': list(indices), 'present': True}
