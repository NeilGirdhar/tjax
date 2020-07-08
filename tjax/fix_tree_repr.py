import importlib
from typing import Any, List

import colorful as cf
from arpeggio import OneOrMore, Optional, ParserPython, RegExMatch, ZeroOrMore
from jaxlib.pytree import PyTreeDef

__all__: List[str] = []


# Grammar
def g_tree() -> Any:
    return 'PyTreeDef(', g_type, ', ', g_values, ')'
def g_type() -> Any:
    return [g_dict, g_tuple, g_none, g_class]
def g_none() -> Any:
    return 'None'
def g_tuple() -> Any:
    return 'tuple'
def g_class() -> Any:
    return ("<class ", g_key, ">", g_crap)
def g_dict() -> Any:
    return ('dict[[', g_key, ZeroOrMore(',', g_key), ']]')
def g_crap() -> Any:
    return "[", RegExMatch(r"[^\[\]]*"), ZeroOrMore(g_crap), RegExMatch(r"[^\[\]]*"), "]"
def g_key() -> Any:
    return [RegExMatch("'[^']*'"),
            ('(', g_key, OneOrMore(', ', g_key), ')')]
def g_values() -> Any:
    return '[', Optional(g_value), ZeroOrMore(',', g_value), ']'
def g_value() -> Any:
    return [g_star, g_tree]
def g_star() -> Any:
    return '*'


indentation = 4


def display(r: str, result: Any, indent: int) -> str:
    def get(node: Any) -> str:
        return r[node.position: node.position_end]
    def unquote(s: str) -> str:
        quotes = "\"'"
        if s[0] not in quotes or s[-1] not in quotes:
            return s
        return s[1: -1]
    t = result[1][0]
    v = result[3]
    rn = t.rule_name[2:]
    keys = None
    if rn == 'class':
        class_qualname = unquote(get(t[1]))
        index = class_qualname.rfind('.')
        class_path = class_qualname[:index]
        class_name = class_qualname[index + 1:]
        module = importlib.import_module(class_path)
        this_class = getattr(module, class_name)
        if hasattr(this_class, 'tree_fields'):
            keys = this_class.tree_fields
        value_type = cf.blue(f"{class_name}")
    elif rn == 'dict':
        keys = [unquote(get(x)) for x in t.g_key]
        value_type = cf.yellow("dict")
    elif rn == 'tuple':
        value_type = cf.cyan("tuple")
    elif rn == 'none':
        value_type = cf.base1("None")
    else:
        assert False
    retval = value_type + "\n"
    if len(v) > 2:
        for i, x in enumerate(v.g_value):
            if x[0].rule_name == 'g_star':
                # retval += " " * (indent + indentation) + cf.base0('*') + "\n"
                continue
            retval += " " * (indent + indentation)
            if keys is not None:
                retval += cf.magenta(keys[i]) + cf.base00("=")
            retval += display(r, x[0], indent + indentation)
    return str(retval)


old_repr = PyTreeDef.__repr__


def new_repr(self: PyTreeDef) -> str:
    r = old_repr(self)
    parser = ParserPython(g_tree)
    try:
        result = parser.parse(r)
        return "\n" + display(r, result, 0)
    except:
        return r


PyTreeDef.__repr__ = new_repr
