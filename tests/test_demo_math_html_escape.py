"""Math lifted out of model text must not bypass the HTML sanitizer.

``sanitize_message_content`` feeds ``st.markdown(unsafe_allow_html=True)`` in
the trace and subagent panels, where model and tool text is shown. Markup
smuggled inside ``$...$`` would run in the Streamlit origin, which holds the
auth token in localStorage.
"""

import pytest

import demo


@pytest.mark.parametrize(
    "content",
    [
        "see $<img src=x onerror=alert(1)>$ here",
        "$$<svg onload=alert(1)></svg>$$",
        "$&lt;img src=x onerror=alert(1)&gt;$",
    ],
)
def test_markup_inside_math_is_escaped(content):
    rendered = demo.sanitize_message_content(content)

    assert "<img" not in rendered
    assert "<svg" not in rendered
    assert "onerror=alert(1)>" not in rendered


def test_plain_math_keeps_its_delimiters():
    rendered = demo.sanitize_message_content("area $a < b$ done")

    assert "\\(a &lt; b\\)" in rendered
