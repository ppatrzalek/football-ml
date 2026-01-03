"""Utility functions for preprocessors."""


def time_to_seconds(time_str):
    if time_str is None:
        return 90 * 60  # 120 minutes = 7200 seconds
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s