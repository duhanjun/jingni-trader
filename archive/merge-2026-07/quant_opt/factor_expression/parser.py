"""
Pratt top-down parser for the factor expression DSL.

Grammar (informal)
------------------

::

    expr      := term (('+' | '-') term)*
    term      := factor (('*' | '/') factor)*
    factor    := unary
    unary     := '-' unary | '+' unary | atom
    atom      := number
               | column                  # $open, $close, Close, ...
               | identifier '(' args ')'  # function call
               | '(' expr ')'
    args      := expr (',' expr)*

Supported tokens
----------------

* **numbers** — ``1``, ``0.5``, ``1e-3`` ...
* **columns** — bare identifiers (``Close``, ``Volume``) *or* prefixed
  with ``$`` (``$close``); the engine treats them as case-insensitive
  lookups against the input frame.
* **operators** — ``+ - * /`` and unary ``- +``
* **function calls** — ``MA(Close, 20)``, ``CsRank($volume)`` ...

The parser is intentionally small: ~120 lines, no third-party deps.
It returns a :class:`Node` tree that the engine walks recursively to
materialise results.  ``Node`` is a plain ``dataclass`` so it is easy
to inspect in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


# ── AST nodes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Node:
    """Recursive AST node.  ``op`` is ``None`` for leaves."""

    op: Optional[str]
    args: Tuple["Node", ...] = ()
    name: Optional[str] = None
    value: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        if self.op is None:
            if self.name is not None:
                return f"${self.name}" if self.name.startswith("$") else self.name
            return str(self.value)
        if len(self.args) == 1:
            return f"{self.op}({self.args[0]!r})"
        if self.op in {"+", "-", "*", "/"}:
            return f"({self.args[0]!r} {self.op} {self.args[1]!r})"
        return f"{self.op}({', '.join(repr(a) for a in self.args)})"


# ── Tokeniser ────────────────────────────────────────────────────────


_TOKEN_PATTERNS: List[Tuple[str, str]] = [
    ("NUMBER", r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"),
    ("IDENT", r"\$?[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"[+\-*/()]"),
    ("COMMA", r","),
    ("WS", r"\s+"),
]

_TOKEN_RE = __import__("re").compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_PATTERNS)
)


@dataclass
class Token:
    kind: str
    text: str
    pos: int


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(f"Unexpected character at {pos}: {text[pos]!r}")
        kind = m.lastgroup
        if kind == "WS":
            pos = m.end()
            continue
        tokens.append(Token(kind, m.group(), m.start()))
        pos = m.end()
    tokens.append(Token("EOF", "", len(text)))
    return tokens


# ── Pratt parser ─────────────────────────────────────────────────────


class _Parser:
    """Pratt parser implementation."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.i = 0

    # helpers --------------------------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[self.i + offset]

    def eat(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def expect(self, kind: str, text: Optional[str] = None) -> Token:
        tok = self.eat()
        if tok.kind != kind or (text is not None and tok.text != text):
            raise ValueError(
                f"Expected {kind} {text or ''!r} but got {tok.kind} {tok.text!r}"
            )
        return tok

    # precedence -----------------------------------------------------------

    # binding power: higher = tighter binding
    _BP = {
        "+": (10, 11),
        "-": (10, 11),
        "*": (20, 21),
        "/": (20, 21),
        "u-": (30, 30),  # unary minus
        "u+": (30, 30),  # unary plus
    }

    def parse(self) -> Node:
        node = self.expr(0)
        if self.peek().kind != "EOF":
            raise ValueError(f"Unexpected token at end: {self.peek().text!r}")
        return node

    def expr(self, min_bp: int) -> Node:
        # prefix / atom
        tok = self.eat()
        if tok.kind == "NUMBER":
            left = Node(op=None, value=float(tok.text))
        elif tok.kind == "IDENT":
            if self.peek().kind == "OP" and self.peek().text == "(":
                self.eat()  # consume '('
                args: List[Node] = []
                if not (self.peek().kind == "OP" and self.peek().text == ")"):
                    while True:
                        args.append(self.expr(0))
                        if self.peek().kind == "COMMA":
                            self.eat()
                            continue
                        break
                self.expect("OP", ")")
                left = Node(op=tok.text, args=tuple(args))
            else:
                left = Node(op=None, name=tok.text)
        elif tok.kind == "OP" and tok.text == "(":
            left = self.expr(0)
            self.expect("OP", ")")
        elif tok.kind == "OP" and tok.text in {"-", "+"}:
            bp_l, bp_r = self._BP["u-" if tok.text == "-" else "u+"]
            operand = self.expr(bp_l)
            left = Node(op=tok.text, args=(operand,))
        else:
            raise ValueError(f"Unexpected token: {tok.text!r}")

        # infix
        while True:
            tok = self.peek()
            if tok.kind != "OP" or tok.text not in self._BP:
                break
            bp_l, bp_r = self._BP[tok.text]
            if bp_l < min_bp:
                break
            self.eat()
            right = self.expr(bp_r)
            left = Node(op=tok.text, args=(left, right))

        return left


def parse(text: str) -> Node:
    """Parse a factor expression and return the AST."""
    tokens = tokenize(text)
    return _Parser(tokens).parse()
