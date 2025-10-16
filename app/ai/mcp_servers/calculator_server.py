from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a"""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns error if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent"""
    return base**exponent


if __name__ == "__main__":
    mcp.run(transport="stdio")
