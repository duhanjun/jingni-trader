"""
Expression DSL - Tokenizer
==========================

A lightweight lexical scanner for the Qlib-inspired factor expression DSL.

Supported token types
---------------------
- ``FIELD``   : ``$open``, ``$close``, ``$volume`` ...  (a field reference)
- ``NUMBER``  : integer or float literal
- ``IDENT``   : function name, e.g. ``Mean``, ``Ref``, ``Std``
- ``LPAREN``  : ``(``
- ``RPAREN``  : ``)``
- ``COMMA``   : ``,``
- ``OP``      : ``+ - * /``
- ``EOF``     : end of input

Design notes
------------
The DSL is intentionally small: only the syntactic features needed to
express typical A-share alpha factors.  It is meant to be the
foundation of a "factor in one line" workflow similar to Qlib's
expression engine.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Iterator


class TokenType(str, Enum):
    FIELD = "FIELD"
    NUMBER = "NUMBER"
    IDENT = "IDENT"
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    COMMA = "COMMA"
    OP = "OP"
    EOF = "EOF"


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, @ {self.pos})"


class TokenizeError(ValueError):
    pass


def tokenize(expr: str) -> List[Token]:
    """Convert an expression string into a list of :class:`Token`.

    Parameters
    ----------
    expr : str
        A factor expression such as ``"Mean($close, 20) / $close"``.

    Returns
    -------
    list[Token]
        Token list ending with :class:`TokenType.EOF`.
    """
    return list(_iter_tokens(expr))


def _iter_tokens(expr: str) -> Iterator[Token]:
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "$":
            j = i + 1
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            if j == i + 1:
                raise TokenizeError(f"字段名缺失: 位置 {i}")
            yield Token(TokenType.FIELD, expr[i + 1:j], i)
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            yield Token(TokenType.IDENT, expr[i:j], i)
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and expr[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < n and (expr[j].isdigit() or (expr[j] == "." and not seen_dot)):
                if expr[j] == ".":
                    seen_dot = True
                j += 1
            yield Token(TokenType.NUMBER, expr[i:j], i)
            i = j
            continue
        if ch == "(":
            yield Token(TokenType.LPAREN, "(", i)
            i += 1
            continue
        if ch == ")":
            yield Token(TokenType.RPAREN, ")", i)
            i += 1
            continue
        if ch == ",":
            yield Token(TokenType.COMMA, ",", i)
            i += 1
            continue
        if ch in "+-*/^":
            yield Token(TokenType.OP, ch, i)
            i += 1
            continue
        raise TokenizeError(f"无法识别的字符 {ch!r} (位置 {i})")
    yield Token(TokenType.EOF, "", n)


if __name__ == "__main__":  # 手动调试
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "Mean($close, 20) / $close"
    for t in tokenize(src):
        print(t)