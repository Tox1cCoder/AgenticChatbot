from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


def _compact(result: float) -> float | int:
    # ``float.is_integer`` is False for inf and nan, where ``int()`` would raise.
    return int(result) if isinstance(result, float) and result.is_integer() else result


@mcp.tool()
def add(a: float, b: float) -> float | int:
    """Add two numbers together"""
    return _compact(a + b)


@mcp.tool()
def subtract(a: float, b: float) -> float | int:
    """Subtract b from a"""
    return _compact(a - b)


@mcp.tool()
def multiply(a: float, b: float) -> float | int:
    """Multiply two numbers together"""
    return _compact(a * b)


@mcp.tool()
def divide(a: float, b: float) -> float | int:
    """Divide a by b. Returns error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return _compact(a / b)


@mcp.tool()
def power(base: float, exponent: float) -> float | int:
    """Raise base to the power of exponent"""
    return _compact(base**exponent)


if __name__ == "__main__":
    mcp.run(transport="stdio")
