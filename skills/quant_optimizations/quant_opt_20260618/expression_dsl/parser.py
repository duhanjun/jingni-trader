"""
Expression DSL - Parser
=======================

Recursive descent parser for the factor expression DSL.

Grammar (informal)
------------------
::

    expr   := term (( "+" | "-" ) term)*
    term   := factor (( "*" | "/" ) factor)*
    factor := FIELD
            | NUMBER
            | IDENT "(" (expr ("," expr)*)? ")"
            | "(" expr ")"

The parser returns an :class:`Ast` (a small dataclass hierarchy) which
is later evaluated by :mod:`expression_dsl.evaluator`.

This module is intentionally framework-free so it can be reused from
Jupyter / scripts / web UI.
"""
from dataclasses import dataclass
from typing import List, Optional, Union

from .tokenizer import Token, TokenType, tokenize


# --- AST node types --------------------------------------------------------


@dataclass
class AstNode:
    pass


@dataclass
class FieldNode(AstNode):
    name: str  # without leading '$'


@dataclass
class NumberNode(AstNode):
    value: float


@dataclass
class BinaryOpNode(AstNode):
    op: str
    left: AstNode
    right: AstNode


@dataclass
class UnaryOpNode(AstNode):
    op: str
    operand: AstNode


@dataclass
class CallNode(AstNode):
    func: str
    args: List[AstNode]


# --- Parser ----------------------------------------------------------------


class ParseError(ValueError):
    pass


class _Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[self.pos + offset]

    def consume(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, ttype: TokenType, hint: str = "") -> Token:
        tok = self.peek()
        if tok.type is not ttype:
            raise ParseError(
                f"期望 {ttype.name} 但得到 {tok.type.name} ({tok.value!r})"
                f" @ {tok.pos}{(' 提示: ' + hint) if hint else ''}"
            )
        return self.consume()

    # --- grammar rules ---

    def parse(self) -> AstNode:
        node = self.expr()
        if self.peek().type is not TokenType.EOF:
            raise ParseError(f"表达式末尾有多余 token: {self.peek()}")
        return node

    def expr(self) -> AstNode:
        node = self.term()
        while self.peek().type is TokenType.OP and self.peek().value in ("+", "-"):
            op = self.consume().value
            right = self.term()
            node = BinaryOpNode(op, node, right)
        return node

    def term(self) -> AstNode:
        node = self.power()
        while self.peek().type is TokenType.OP and self.peek().value in ("*", "/"):
            op = self.consume().value
            right = self.power()
            node = BinaryOpNode(op, node, right)
        return node

    def power(self) -> AstNode:
        node = self.factor()
        if self.peek().type is TokenType.OP and self.peek().value == "^":
            self.consume()
            right = self.power()  # right-associative
            node = BinaryOpNode("^", node, right)
        return node

    def factor(self) -> AstNode:
        tok = self.peek()
        if tok.type is TokenType.OP and tok.value == "-":
            self.consume()
            return UnaryOpNode("-", self.factor())
        if tok.type is TokenType.OP and tok.value == "+":
            self.consume()
            return self.factor()
        if tok.type is TokenType.LPAREN:
            self.consume()
            node = self.expr()
            self.expect(TokenType.RPAREN, "括号不匹配")
            return node
        if tok.type is TokenType.FIELD:
            self.consume()
            return FieldNode(tok.value)
        if tok.type is TokenType.NUMBER:
            self.consume()
            return NumberNode(float(tok.value))
        if tok.type is TokenType.IDENT:
            return self._call()
        raise ParseError(f"无法解析 token: {tok}")

    def _call(self) -> CallNode:
        name_tok = self.expect(TokenType.IDENT, "函数名后应接 (")
        self.expect(TokenType.LPAREN)
        args: List[AstNode] = []
        if self.peek().type is not TokenType.RPAREN:
            args.append(self.expr())
            while self.peek().type is TokenType.COMMA:
                self.consume()
                args.append(self.expr())
        self.expect(TokenType.RPAREN, "缺少右括号")
        return CallNode(name_tok.value, args)


def parse(expr: str) -> AstNode:
    """Parse a DSL string into an AST."""
    return _Parser(tokenize(expr)).parse()


if __name__ == "__main__":  # 手动调试
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "Corr(Std($close, 5), Mean($volume, 10), 20)"
    print(parse(src))