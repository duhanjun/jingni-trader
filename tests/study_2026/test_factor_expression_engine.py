"""
因子表达式引擎验证测试
============================
借鉴来源: Microsoft Qlib (https://github.com/microsoft/qlib)
         - Qlib Expression Engine: DSL-based factor definition
         - 文档: https://qlib.readthedocs.io/en/latest/component/data.html
优化方向: 因子库可扩展性 - 用声明式 DSL 替代硬编码因子计算，
         降低 LLM 生成因子的复杂度，从"生成完整 Python 代码"降级为"生成简单数学表达式"
验证内容:
  1. 表达式解析器：将字符串表达式解析为 AST 计算树
  2. 常用运算符：算术、比较、逻辑、滚动窗口
  3. 因子计算：将 AST 作用于行情 DataFrame 得到因子值
  4. LLM 友好性：对比 Python 代码 vs DSL 表达式长度
  5. 边界条件：非法表达式、缺失列、空数据
"""

import unittest
import re
import operator
from typing import Dict, List, Any, Optional, Callable, Union
import numpy as np
import pandas as pd


# ============================================================
# 因子表达式引擎实现（参考 Qlib Expression Engine 设计）
# ============================================================

class FactorExprError(Exception):
    """表达式引擎异常"""
    pass


class Token:
    """词法分析 Token"""
    def __init__(self, type_: str, value: str, pos: int = 0):
        self.type = type_
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"Token({self.type}, {self.value})"


class ASTNode:
    """抽象语法树节点基类"""
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class FieldNode(ASTNode):
    """行情字段节点，如 $close, $open, $volume"""
    def __init__(self, field: str):
        self.field = field

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        col_name = self.field.lower()
        if col_name not in data.columns:
            raise FactorExprError(f"字段 '{self.field}' 不存在，可用字段: {list(data.columns)}")
        return data[col_name]

    def __repr__(self):
        return f"Field({self.field})"


class ConstNode(ASTNode):
    """常数节点"""
    def __init__(self, value: float):
        self.value = value

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        # 返回与 DataFrame 等长的常数序列
        return pd.Series(self.value, index=data.index)

    def __repr__(self):
        return f"Const({self.value})"


class BinaryOpNode(ASTNode):
    """二元运算节点"""
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        left_val = self.left.evaluate(data)
        right_val = self.right.evaluate(data)

        # 如果 right_val 是标量 Series，广播；否则对齐索引
        ops = {
            '+': operator.add, '-': operator.sub,
            '*': operator.mul, '/': operator.truediv,
            '>': operator.gt, '<': operator.lt,
            '>=': operator.ge, '<=': operator.le,
            '==': operator.eq, '!=': operator.ne,
            '&': operator.and_, '|': operator.or_,
        }
        if self.op not in ops:
            raise FactorExprError(f"不支持的运算符: {self.op}")
        return ops[self.op](left_val, right_val)

    def __repr__(self):
        return f"BinOp({self.op}, {self.left}, {self.right})"


class RollingOpNode(ASTNode):
    """滚动窗口运算节点，如 MA(close, 20), Std(close, 5)"""
    def __init__(self, func_name: str, child: ASTNode, window: int):
        self.func_name = func_name
        self.child = child
        self.window = window

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        val = self.child.evaluate(data)
        if self.func_name == 'MA':
            return val.rolling(window=self.window, min_periods=1).mean()
        elif self.func_name == 'STD':
            return val.rolling(window=self.window, min_periods=1).std()
        elif self.func_name == 'SUM':
            return val.rolling(window=self.window, min_periods=1).sum()
        elif self.func_name == 'MAX':
            return val.rolling(window=self.window, min_periods=1).max()
        elif self.func_name == 'MIN':
            return val.rolling(window=self.window, min_periods=1).min()
        else:
            raise FactorExprError(f"不支持的滚动函数: {self.func_name}")

    def __repr__(self):
        return f"Rolling({self.func_name}, {self.child}, {self.window})"


class RefNode(ASTNode):
    """时序位移节点，如 Ref(close, 1) = 前一日收盘价"""
    def __init__(self, child: ASTNode, shift: int):
        self.child = child
        self.shift = shift

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        val = self.child.evaluate(data)
        return val.shift(self.shift)

    def __repr__(self):
        return f"Ref({self.child}, {self.shift})"


class IfNode(ASTNode):
    """条件表达式: If(condition, true_expr, false_expr)"""
    def __init__(self, condition: ASTNode, true_expr: ASTNode, false_expr: ASTNode):
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        cond = self.condition.evaluate(data)
        true_val = self.true_expr.evaluate(data)
        false_val = self.false_expr.evaluate(data)
        return pd.Series(np.where(cond.values, true_val.values, false_val.values),
                         index=data.index)

    def __repr__(self):
        return f"If({self.condition}, {self.true_expr}, {self.false_expr})"


class ExpressionParser:
    """
    因子表达式解析器

    支持的语法:
      - 字段: $close, $open, $high, $low, $volume, $amount
      - 常数: 数字（整数或小数）
      - 算术: +, -, *, /
      - 比较: >, <, >=, <=, ==, !=
      - 逻辑: &, |
      - 函数: MA(expr, N), STD(expr, N), SUM(expr, N), MAX(expr, N), MIN(expr, N)
      - 时序: Ref(expr, N)  - N正数为前移, N负数为后移
      - 条件: If(cond, true, false)
      - 括号: () 控制优先级

    运算符优先级（从低到高）:
      1. | 逻辑或
      2. & 逻辑与
      3. > < >= <= == != 比较
      4. + - 加减
      5. * / 乘除
    """

    FUNC_NAMES = {'MA', 'STD', 'SUM', 'MAX', 'MIN', 'Ref', 'If'}

    def __init__(self):
        self.tokens: List[Token] = []
        self.pos: int = 0

    def _tokenize(self, expr: str) -> List[Token]:
        """词法分析：将表达式字符串拆分为 Token 列表"""
        tokens = []
        i = 0
        expr = expr.strip()

        while i < len(expr):
            ch = expr[i]

            # 跳过空白
            if ch.isspace():
                i += 1
                continue

            # 字段引用 $xxx
            if ch == '$':
                match = re.match(r'\$([a-zA-Z_]\w*)', expr[i:])
                if match:
                    tokens.append(Token('FIELD', match.group(0), i))
                    i += len(match.group(0))
                    continue

            # 数字
            if ch.isdigit() or (ch == '.' and i + 1 < len(expr) and expr[i + 1].isdigit()):
                match = re.match(r'\d+\.?\d*', expr[i:])
                if match:
                    tokens.append(Token('NUMBER', match.group(0), i))
                    i += len(match.group(0))
                    continue

            # 多字符运算符
            if ch in '>=<!':
                if i + 1 < len(expr) and expr[i + 1] == '=':
                    tokens.append(Token('OP', ch + '=', i))
                    i += 2
                    continue
                elif i + 1 < len(expr) and expr[i + 1] == ch and ch in '<>':
                    # 理论上 != 已处理，这里只有 <>
                    pass

            # 单字符运算符
            if ch in '+-*/':
                tokens.append(Token('OP', ch, i))
                i += 1
                continue

            if ch == '&':
                tokens.append(Token('OP', '&', i))
                i += 1
                continue
            elif ch == '|':
                tokens.append(Token('OP', '|', i))
                i += 1
                continue

            # 比较运算符
            if ch in '><':
                tokens.append(Token('OP', ch, i))
                i += 1
                continue

            if ch == '=' and i + 1 < len(expr) and expr[i + 1] == '=':
                tokens.append(Token('OP', '==', i))
                i += 2
                continue

            if ch == '!' and i + 1 < len(expr) and expr[i + 1] == '=':
                tokens.append(Token('OP', '!=', i))
                i += 2
                continue

            # 括号
            if ch == '(':
                tokens.append(Token('LPAREN', '(', i))
                i += 1
                continue
            if ch == ')':
                tokens.append(Token('RPAREN', ')', i))
                i += 1
                continue

            # 逗号
            if ch == ',':
                tokens.append(Token('COMMA', ',', i))
                i += 1
                continue

            # 标识符（函数名）
            match = re.match(r'[a-zA-Z_]\w*', expr[i:])
            if match:
                name = match.group(0)
                # 看后面是否为 '(' 来确定是函数调用
                j = i + len(name)
                while j < len(expr) and expr[j].isspace():
                    j += 1
                if j < len(expr) and expr[j] == '(':
                    tokens.append(Token('FUNC', name, i))
                else:
                    # 不是函数，可能是字段无 $ 前缀的写法
                    tokens.append(Token('FIELD', name, i))
                i += len(name)
                continue

            raise FactorExprError(f"无法识别的字符 '{ch}' 在位置 {i}")

        return tokens

    def _peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> Optional[Token]:
        token = self._peek()
        if token:
            self.pos += 1
        return token

    def _expect(self, type_: str) -> Token:
        token = self._consume()
        if token is None or token.type != type_:
            raise FactorExprError(f"期望类型 {type_}, 实际 {token}")
        return token

    def parse(self, expr: str) -> ASTNode:
        """解析因子表达式，返回 AST 根节点"""
        self.tokens = self._tokenize(expr)
        self.pos = 0
        if not self.tokens:
            raise FactorExprError("空表达式")
        ast = self._parse_or()
        if self.pos < len(self.tokens):
            raise FactorExprError(f"多余的 Token: {self.tokens[self.pos]}")
        return ast

    # --- 递归下降解析 ---

    def _parse_or(self) -> ASTNode:
        """解析逻辑或 |"""
        node = self._parse_and()
        while self._peek() and self._peek().type == 'OP' and self._peek().value == '|':
            self._consume()
            node = BinaryOpNode('|', node, self._parse_and())
        return node

    def _parse_and(self) -> ASTNode:
        """解析逻辑与 &"""
        node = self._parse_compare()
        while self._peek() and self._peek().type == 'OP' and self._peek().value == '&':
            self._consume()
            node = BinaryOpNode('&', node, self._parse_compare())
        return node

    def _parse_compare(self) -> ASTNode:
        """解析比较运算"""
        node = self._parse_add()
        compare_ops = {'>', '<', '>=', '<=', '==', '!='}
        while self._peek() and self._peek().type == 'OP' and self._peek().value in compare_ops:
            op = self._consume().value
            node = BinaryOpNode(op, node, self._parse_add())
        return node

    def _parse_add(self) -> ASTNode:
        """解析加减"""
        node = self._parse_mul()
        while self._peek() and self._peek().type == 'OP' and self._peek().value in ('+', '-'):
            op = self._consume().value
            node = BinaryOpNode(op, node, self._parse_mul())
        return node

    def _parse_mul(self) -> ASTNode:
        """解析乘除"""
        node = self._parse_unary()
        while self._peek() and self._peek().type == 'OP' and self._peek().value in ('*', '/'):
            op = self._consume().value
            node = BinaryOpNode(op, node, self._parse_unary())
        return node

    def _parse_unary(self) -> ASTNode:
        """解析一元运算符 (- 目前仅支持取负)"""
        if self._peek() and self._peek().type == 'OP' and self._peek().value == '-':
            self._consume()
            child = self._parse_unary()
            return BinaryOpNode('*', ConstNode(-1), child)
        return self._parse_atom()

    def _parse_atom(self) -> ASTNode:
        """解析原子表达式：字段、数字、函数调用、括号"""
        token = self._peek()
        if token is None:
            raise FactorExprError("表达式不完整")

        # 字段引用 $close
        if token.type == 'FIELD':
            self._consume()
            return FieldNode(token.value[1:] if token.value.startswith('$') else token.value)

        # 函数调用: MA(expr, N), Ref(expr, N), If(cond, t, f)
        if token.type == 'FUNC':
            func_name = self._consume().value
            self._expect('LPAREN')

            if func_name == 'If':
                cond = self._parse_or()
                self._expect('COMMA')
                true_expr = self._parse_or()
                self._expect('COMMA')
                false_expr = self._parse_or()
                self._expect('RPAREN')
                return IfNode(cond, true_expr, false_expr)
            elif func_name == 'Ref':
                child = self._parse_or()
                self._expect('COMMA')
                # support negative shift like Ref($close, -1)
                shift = 0
                negate = False
                if self._peek() and self._peek().type == 'OP' and self._peek().value == '-':
                    self._consume()
                    negate = True
                shift_token = self._expect('NUMBER')
                shift = int(float(shift_token.value))
                if negate:
                    shift = -shift
                self._expect('RPAREN')
                return RefNode(child, shift)
            elif func_name in {'MA', 'STD', 'SUM', 'MAX', 'MIN'}:
                child = self._parse_or()
                self._expect('COMMA')
                window_token = self._expect('NUMBER')
                window = int(float(window_token.value))
                self._expect('RPAREN')
                return RollingOpNode(func_name, child, window)
            else:
                raise FactorExprError(f"未知函数: {func_name}")

        # 数字
        if token.type == 'NUMBER':
            self._consume()
            return ConstNode(float(token.value))

        # 括号
        if token.type == 'LPAREN':
            self._consume()
            node = self._parse_or()
            self._expect('RPAREN')
            return node

        raise FactorExprError(f"意外的 Token: {token}")


# ============================================================
# 辅助：生成表达式与 Python 代码对比
# ============================================================

def compare_expression_complexity(dsl_expr: str, python_code: str) -> Dict[str, int]:
    """对比 DSL 表达式和 Python 代码的复杂度"""
    return {
        "dsl_chars": len(dsl_expr),
        "dsl_tokens": len(dsl_expr.split()),
        "python_chars": len(python_code.strip()),
        "python_lines": len(python_code.strip().split('\n')),
        "complexity_reduction": round((1 - len(dsl_expr) / len(python_code.strip())) * 100, 1)
    }


# ============================================================
# 测试用例
# ============================================================

class TestExpressionParser(unittest.TestCase):
    """表达式解析器测试"""

    def setUp(self):
        self.parser = ExpressionParser()

    def test_simple_field(self):
        """简单字段解析"""
        ast = self.parser.parse("$close")
        self.assertIsInstance(ast, FieldNode)
        self.assertEqual(ast.field, "close")

    def test_arithmetic(self):
        """算术表达式"""
        ast = self.parser.parse("$high - $low")
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, "-")

    def test_rolling_function(self):
        """滚动函数"""
        ast = self.parser.parse("MA($close, 20)")
        self.assertIsInstance(ast, RollingOpNode)
        self.assertEqual(ast.func_name, "MA")
        self.assertEqual(ast.window, 20)

    def test_ref_function(self):
        """时序位移"""
        ast = self.parser.parse("Ref($close, 1)")
        self.assertIsInstance(ast, RefNode)
        self.assertEqual(ast.shift, 1)

    def test_complex_expression(self):
        """复杂嵌套表达式: MACD"""
        expr = '(EMA($close, 12) - EMA($close, 26))/$close'
        # EMA 暂未实现，改用 MA 测试嵌套
        expr = '(MA($close, 12) - MA($close, 26))/$close'
        ast = self.parser.parse(expr)
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, '/')

    def test_conditional_if(self):
        """条件表达式"""
        ast = self.parser.parse("If($close > MA($close, 20), $close, 0)")
        self.assertIsInstance(ast, IfNode)

    def test_precedence(self):
        """运算符优先级: 乘法优先于加法"""
        ast = self.parser.parse("$close + $high * 0.5")
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, '+')
        # 右侧应为乘法
        self.assertIsInstance(ast.right, BinaryOpNode)
        self.assertEqual(ast.right.op, '*')

    def test_parse_error_invalid(self):
        """非法表达式应抛出异常"""
        with self.assertRaises(FactorExprError):
            self.parser.parse("")

    def test_parse_error_unknown_func(self):
        """未知函数应抛出异常"""
        with self.assertRaises(FactorExprError):
            self.parser.parse("UnknownFunc($close, 10)")

    def test_parse_negative_number(self):
        """负号处理"""
        ast = self.parser.parse("$close - 10")
        self.assertIsInstance(ast, BinaryOpNode)


class TestASTEvaluation(unittest.TestCase):
    """AST 求值测试"""

    def setUp(self):
        self.parser = ExpressionParser()
        # 构造模拟行情数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='B')
        self.data = pd.DataFrame({
            'open': np.random.uniform(10, 50, 200),
            'high': np.random.uniform(11, 52, 200),
            'low': np.random.uniform(9, 49, 200),
            'close': np.random.uniform(10, 51, 200),
            'volume': np.random.uniform(1e5, 1e7, 200),
            'amount': np.random.uniform(1e6, 1e8, 200),
        }, index=dates)
        # 确保 high >= low
        self.data['low'] = self.data[['high', 'low']].min(axis=1) - 0.1
        self.data['high'] = self.data[['high', 'low']].max(axis=1) + 0.1

    def test_field_eval(self):
        """字段求值"""
        ast = self.parser.parse("$close")
        result = ast.evaluate(self.data)
        pd.testing.assert_series_equal(result, self.data['close'])

    def test_arithmetic_eval(self):
        """算术求值"""
        ast = self.parser.parse("$high - $low")
        result = ast.evaluate(self.data)
        expected = self.data['high'] - self.data['low']
        pd.testing.assert_series_equal(result, expected)

    def test_ma_eval(self):
        """移动平均求值"""
        window = 10
        ast = self.parser.parse(f"MA($close, {window})")
        result = ast.evaluate(self.data)
        expected = self.data['close'].rolling(window=window, min_periods=1).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_ref_eval(self):
        """时序位移求值"""
        ast = self.parser.parse("Ref($close, 1)")
        result = ast.evaluate(self.data)
        expected = self.data['close'].shift(1)
        pd.testing.assert_series_equal(result, expected)

    def test_complex_factor_eval(self):
        """复杂因子: (MA(close,5)-MA(close,20))/MA(close,20)"""
        expr = "(MA($close, 5) - MA($close, 20)) / MA($close, 20)"
        ast = self.parser.parse(expr)
        result = ast.evaluate(self.data)

        ma5 = self.data['close'].rolling(5, min_periods=1).mean()
        ma20 = self.data['close'].rolling(20, min_periods=1).mean()
        expected = (ma5 - ma20) / ma20

        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_conditional_eval(self):
        """条件表达式求值: 如果收盘价>均线则返回1，否则0"""
        ast = self.parser.parse("If($close > MA($close, 20), 1, 0)")
        result = ast.evaluate(self.data)

        ma20 = self.data['close'].rolling(20, min_periods=1).mean()
        expected = pd.Series(
            np.where(self.data['close'].values > ma20.values, 1.0, 0.0),
            index=self.data.index
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_logical_and_eval(self):
        """逻辑与"""
        ast = self.parser.parse("($close > MA($close, 20)) & ($volume > MA($volume, 5))")
        result = ast.evaluate(self.data)
        self.assertEqual(result.dtype, bool)

    def test_missing_field_error(self):
        """缺失字段求值应报错"""
        ast = self.parser.parse("$nonexistent")
        with self.assertRaises(FactorExprError):
            ast.evaluate(self.data)

    def test_empty_data(self):
        """空数据求值"""
        ast = self.parser.parse("$close")
        empty_data = pd.DataFrame(columns=['close'])
        result = ast.evaluate(empty_data)
        self.assertEqual(len(result), 0)

    def test_single_row_data(self):
        """单行数据求值"""
        ast = self.parser.parse("MA($close, 20)")
        single_data = pd.DataFrame({'close': [10.0]})
        result = ast.evaluate(single_data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0], 10.0)


class TestLLMFriendliness(unittest.TestCase):
    """LLM 友好性对比测试"""

    def setUp(self):
        self.parser = ExpressionParser()

    def test_macd_complexity_comparison(self):
        """MACD 表达式 vs Python 代码复杂度对比"""
        dsl = "(MA($close, 12) - MA($close, 26)) / $close - MA((MA($close, 12) - MA($close, 26)) / $close, 9) / $close"

        python_code = """
import pandas as pd
ema12 = data['close'].ewm(span=12, adjust=False).mean()
ema26 = data['close'].ewm(span=26, adjust=False).mean()
dif = (ema12 - ema26) / data['close']
dea = dif.ewm(span=9, adjust=False).mean() / data['close']
macd = dif - dea
""".strip()

        comp = compare_expression_complexity(dsl, python_code)
        self.assertLess(comp["dsl_chars"], comp["python_chars"])
        print(f"\n  MACD 因子复杂度对比:")
        print(f"    DSL 表达式: {comp['dsl_chars']} 字符")
        print(f"    Python 代码: {comp['python_chars']} 字符, {comp['python_lines']} 行")
        print(f"    复杂度降低: {comp['complexity_reduction']}%")

    def test_rsi_complexity_comparison(self):
        """RSI 表达式 vs Python 代码复杂度对比"""
        dsl = "MA(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / MA(Abs($close - Ref($close, 1)), 14) * 100"

        python_code = """
import pandas as pd
delta = data['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))
""".strip()

        comp = compare_expression_complexity(dsl, python_code)
        self.assertLess(comp["dsl_chars"], comp["python_chars"])
        print(f"\n  RSI 因子复杂度对比:")
        print(f"    DSL 表达式: {comp['dsl_chars']} 字符")
        print(f"    Python 代码: {comp['python_chars']} 字符, {comp['python_lines']} 行")
        print(f"    复杂度降低: {comp['complexity_reduction']}%")

    def test_llm_generation_simplicity(self):
        """验证 DSL 表达式的 LLM 生成可行性

        核心观点（来自 Qlib 经验）:
        - LLM 生成 DSL 表达式比生成 Python 代码更容易且更可靠
        - DSL 是纯声明式的，不需要处理导入、变量名、代码结构
        - DSL 表达式可以直接作为字符串传递给引擎，无需 exec/eval 的安全风险
        """
        # 模拟 LLM 可能生成的因子表达式
        sample_factors = [
            "$close / MA($close, 60) - 1",                    # 60日收益率
            "($high - $low) / $close",                         # 振幅
            "($close - MA($close, 20)) / STD($close, 20)",    # 标准化的偏离
            "MA($volume, 5) / MA($volume, 20)",               # 量比
            "MAX($high, 20) / $close - 1",                    # 距20日高点距离
        ]

        parser = ExpressionParser()
        for i, expr in enumerate(sample_factors):
            try:
                ast = parser.parse(expr)
                self.assertIsNotNone(ast)
                print(f"    因子 {i+1} 解析成功: {expr}")
            except FactorExprError as e:
                self.fail(f"因子 {i+1} 解析失败: {expr}, 错误: {e}")

        print(f"\n  全部 {len(sample_factors)} 个 DSL 因子解析成功 - LLM 可安全生成并执行")


# ============================================================
# 因子注册表 - 扩展性验证
# ============================================================

class FactorRegistry:
    """
    因子注册表：因子名 -> DSL 表达式 的映射

    设计理念（借鉴 Qlib Alpha158 因子库）:
      - 因子用声明式 DSL 定义，而非硬编码
      - 新增因子只需添加一行 DSL 表达式，无需写新代码
      - 天然适合 LLM 驱动的工作流：LLM 提出新因子 = 生成一个 DSL 表达式
    """

    BUILTIN_FACTORS = {
        # 收益率类
        "ret_1d": "Ref($close, -1) / $close - 1",
        "ret_5d": "Ref($close, -5) / $close - 1",
        "ret_20d": "Ref($close, -20) / $close - 1",

        # 波动率类
        "amplitude": "($high - $low) / Ref($close, 1)",
        "volatility_20d": "STD(Ref($close, -1) / $close - 1, 20)",

        # 均线偏离类
        "bias_5": "$close / MA($close, 5) - 1",
        "bias_20": "$close / MA($close, 20) - 1",
        "bias_60": "$close / MA($close, 60) - 1",

        # 量价类
        "volume_ratio_5": "MA($volume, 5) / MA($volume, 20)",
        "turnover_5d": "MA($volume / Ref($close, 1), 5)",

        # 动量类
        "macd_dif": "(MA($close, 12) - MA($close, 26)) / $close",
        "rsi_14": "MA(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) / (MA(If($close > Ref($close, 1), $close - Ref($close, 1), 0), 14) + MA(If(Ref($close, 1) > $close, Ref($close, 1) - $close, 0), 14))",
    }

    def __init__(self):
        self._factors: Dict[str, str] = dict(self.BUILTIN_FACTORS)
        self._parser = ExpressionParser()

    def register(self, name: str, expr: str):
        """注册新因子表达式"""
        # 验证表达式能否解析
        self._parser.parse(expr)
        self._factors[name] = expr

    def compute(self, name: str, data: pd.DataFrame) -> pd.Series:
        """计算单个因子"""
        if name not in self._factors:
            raise FactorExprError(f"因子 '{name}' 未注册")
        ast = self._parser.parse(self._factors[name])
        return ast.evaluate(data)

    def compute_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """批量计算所有因子"""
        results = {}
        for name in self._factors:
            try:
                results[name] = self.compute(name, data)
            except Exception as e:
                results[name] = f"ERROR: {e}"
        return pd.DataFrame(results)

    def list_factors(self) -> List[str]:
        return list(self._factors.keys())

    def get_expression(self, name: str) -> Optional[str]:
        return self._factors.get(name)


class TestFactorRegistry(unittest.TestCase):
    """因子注册表测试"""

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='B')
        self.data = pd.DataFrame({
            'open': np.random.uniform(10, 50, 200),
            'high': np.random.uniform(11, 52, 200),
            'low': np.random.uniform(9, 49, 200),
            'close': np.random.uniform(10, 51, 200),
            'volume': np.random.uniform(1e5, 1e7, 200),
            'amount': np.random.uniform(1e6, 1e8, 200),
        }, index=dates)

    def test_builtin_factors_all_parse(self):
        """所有内置因子应能正常解析"""
        registry = FactorRegistry()
        for name, expr in registry.BUILTIN_FACTORS.items():
            ast = registry._parser.parse(expr)
            self.assertIsNotNone(ast, f"因子 '{name}' 解析失败: {expr}")

    def test_builtin_factors_compute(self):
        """所有内置因子应能正常计算"""
        registry = FactorRegistry()
        for name in registry.list_factors():
            result = registry.compute(name, self.data)
            self.assertIsInstance(result, pd.Series, f"因子 '{name}' 计算失败")
            self.assertEqual(len(result), len(self.data))

    def test_register_new_factor(self):
        """注册新因子 - 无需写新代码"""
        registry = FactorRegistry()
        # 新增一个 30日均线偏离因子，只需一行表达式
        registry.register("bias_30", "$close / MA($close, 30) - 1")
        self.assertIn("bias_30", registry.list_factors())
        result = registry.compute("bias_30", self.data)
        self.assertEqual(len(result), len(self.data))

    def test_batch_compute(self):
        """批量计算所有因子"""
        registry = FactorRegistry()
        df = registry.compute_all(self.data)
        self.assertEqual(len(df), len(self.data))
        print(f"\n  批量计算 {len(registry.list_factors())} 个因子完成, 结果维度: {df.shape}")

    def test_extensibility(self):
        """可扩展性验证: 新增因子无需修改任何现有代码"""
        registry = FactorRegistry()
        initial_count = len(registry.list_factors())

        # 模拟 LLM 生成新的因子表达式并注册
        new_factors = [
            ("alpha_001", "($close - Ref($close, 1)) / $close * $volume / MA($volume, 20)"),
            ("alpha_002", "STD($close / MA($close, 5), 20)"),
            ("alpha_003", "($high - $low) / MA($close, 10)"),
        ]
        for name, expr in new_factors:
            registry.register(name, expr)

        self.assertEqual(len(registry.list_factors()), initial_count + 3)
        print(f"\n  可扩展性验证: 新增 {len(new_factors)} 个因子只需添加 DSL 表达式，无需修改任何代码")


if __name__ == '__main__':
    unittest.main(verbosity=2)