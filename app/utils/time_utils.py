def safe_min(values, default=None):
    values = [v for v in values if v is not None]
    return min(values) if values else default


def safe_max(values, default=None):
    values = [v for v in values if v is not None]
    return max(values) if values else default