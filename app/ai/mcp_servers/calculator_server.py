from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool()
def add(a: float, b: float) -> float | int:
    """Add two numbers together"""
    result = a + b
    return int(result) if result == int(result) else result


@mcp.tool()
def subtract(a: float, b: float) -> float | int:
    """Subtract b from a"""
    result = a - b
    return int(result) if result == int(result) else result


@mcp.tool()
def multiply(a: float, b: float) -> float | int:
    """Multiply two numbers together"""
    result = a * b
    return int(result) if result == int(result) else result


@mcp.tool()
def divide(a: float, b: float) -> float | int:
    """Divide a by b. Returns error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    result = a / b
    return int(result) if result == int(result) else result


@mcp.tool()
def power(base: float, exponent: float) -> float | int:
    """Raise base to the power of exponent"""
    result = base**exponent
    return int(result) if result == int(result) else result


if __name__ == "__main__":
    mcp.run(transport="stdio")
