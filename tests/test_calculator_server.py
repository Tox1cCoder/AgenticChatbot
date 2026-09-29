import math

from app.ai.mcp_servers import calculator_server as calc


def test_integral_results_are_compacted_to_int():
    assert calc.add(2.0, 3.0) == 5 and isinstance(calc.add(2.0, 3.0), int)
    assert calc.divide(7.0, 2.0) == 3.5


def test_non_finite_results_are_returned_instead_of_crashing():
    """``int(inf)`` raised OverflowError and ``int(nan)`` ValueError."""
    assert calc.multiply(1e308, 10.0) == math.inf
    assert math.isnan(calc.subtract(math.inf, math.inf))
