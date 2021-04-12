import sys
from typing import List

import colorful as cf

__all__: List[str] = []


cf.use_style('solarized')
if not sys.stdout.isatty() or not sys.stderr.isatty():
    # Output is piped or redirected.
    cf.disable()
