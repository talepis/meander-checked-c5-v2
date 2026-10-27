"""Small cross-platform process metrics used by the exact engines."""
from __future__ import annotations


def peak_rss_kib() -> int:
    """Return the process high-water resident set in KiB.

    Unix ``resource.ru_maxrss`` is expressed in KiB on the supported Linux
    target.  Windows exposes the equivalent peak working set through psutil.
    Keeping the conversion here avoids changing any arithmetic code.
    """
    try:
        import resource
    except ImportError:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", info.rss)
        return int(peak) // 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
