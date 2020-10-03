# This patch changes dataclass behavior for all dataclasses:  When a non-defaulted dataclass field
# follows a defaulted field, the behaviour has changed from raising a TypeError to making all
# arguments in __init__ following the defaulted field keyword-only.
# See: https://github.com/python/cpython/pull/17322
# type: ignore
# pylint: disable=protected-access
import dataclasses
import functools as ft
import sys

__all__ = []


@ft.wraps(dataclasses._init_fn)
def _init_fn(fields, frozen, has_post_init, self_name, globals_=None):
    """Build ``__init__`` for a data-class."""
    py_37 = sys.version_info < (3, 7, 6)

    locals_ = {f"_type_{f.name}": f.type for f in fields}
    extra = {
        "MISSING": dataclasses.MISSING,
        "_HAS_DEFAULT_FACTORY": dataclasses._HAS_DEFAULT_FACTORY,
    }
    if py_37:
        assert globals_ is None
        globals_ = extra
    else:
        assert globals_ is not None
        locals_.update(extra)

    body_lines = []
    for f in fields:
        line = dataclasses._field_init(
            f, frozen, globals_ if py_37 else locals_, self_name
        )
        if line:
            body_lines.append(line)
    if has_post_init:
        params_str = ",".join(
            f.name for f in fields if f._field_type is dataclasses._FIELD_INITVAR
        )
        body_line = f"{self_name}.{dataclasses._POST_INIT_NAME}({params_str})"
        body_lines.append(body_line)
    if not body_lines:
        body_lines = ["pass"]

    # Edit: args after defaulted args are keyword-only
    seen_default = False
    keyword_only = False
    args = [self_name]
    for f in fields:
        if f.init:
            has_default = f.default is not dataclasses.MISSING
            has_default_factory = f.default_factory is not dataclasses.MISSING
            if has_default or has_default_factory:
                seen_default = True
            elif seen_default and not keyword_only:
                keyword_only = True
                args.append("*")
            args.append(dataclasses._init_param(f))

    return dataclasses._create_fn(
        "__init__",
        args,
        body_lines,
        locals=locals_,
        globals=globals_,
        return_type=None,
    )


dataclasses._init_fn = _init_fn
