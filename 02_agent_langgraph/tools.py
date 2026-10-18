"""The agent's two tools.

A "tool" is just a normal Python function plus a description. The @tool
decorator turns the function signature and the docstring into a JSON schema
that is sent to the model with every request.

IMPORTANT TEACHING POINT:
The docstring is not a comment - it is the prompt. The model reads it to
decide whether this tool is the right one and what to put in each argument.
A vague docstring produces a tool the model never calls, or calls wrongly.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


# =============================================================== TOOL 1
@tool
def get_current_datetime(timezone: str = "Asia/Karachi") -> str:
    """Get the current date and time right now, in any city or region's timezone.

    Use this whenever the user asks what the date is, what day it is, what the
    time is, how long until some date, or anything that depends on "today" or
    "now". The language model has no clock of its own, so it MUST call this
    tool rather than guessing.

    Args:
        timezone: An IANA timezone name such as "Asia/Karachi", "Asia/Tokyo",
            "Europe/London", "America/New_York" or "UTC". Defaults to
            "Asia/Karachi".

    Returns:
        A human readable date and time string for that timezone.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        # Returning the error as text (instead of raising) lets the model
        # read it and try again with a corrected argument.
        return (
            f"Error: '{timezone}' is not a valid IANA timezone name. "
            "Try a name like 'Asia/Karachi', 'Europe/London' or 'UTC'."
        )

    now = datetime.now(zone)
    return (
        f"{now:%A, %d %B %Y, %H:%M:%S} "
        f"({now.tzname()}, timezone {timezone}, day {now.timetuple().tm_yday} of the year)"
    )


# =============================================================== TOOL 2
# Only these operations are allowed. Never use eval() on text from a model
# or a user - this whitelist is the safe way to do arithmetic.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_evaluate(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


@tool
def calculator(expression: str) -> str:
    """Do exact arithmetic. Use this for ANY calculation, however simple.

    Language models predict text and are unreliable at mental arithmetic, so
    always use this tool instead of working the answer out yourself.

    Args:
        expression: A arithmetic expression in Python syntax, for example
            "1575 / 3 + 12", "22 - 7", "(4500 * 6) + 2000" or "2 ** 10".
            Numbers and the operators + - * / // % ** and brackets only.
            Do not include currency symbols, commas or units.

    Returns:
        The result of the calculation, or an error message explaining what
        was wrong with the expression.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' - {exc}"

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    elif isinstance(result, float):
        result = round(result, 6)
    return f"{expression} = {result}"


# The list the agent binds to the model.
TOOLS = [get_current_datetime, calculator]
