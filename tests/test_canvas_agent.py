from __future__ import annotations

from app.ai.agents.canvas_agent import _extract_artifact


def test_canvas_artifact_response_shape_omits_editable_hint():
    artifact = _extract_artifact(
        """
Built a small page.

```html
<!doctype html>
<html>
  <head><title>Demo Canvas</title></head>
  <body>Hello</body>
</html>
```
"""
    )

    assert artifact == {
        "content": (
            "<!doctype html>\n"
            "<html>\n"
            "  <head><title>Demo Canvas</title></head>\n"
            "  <body>Hello</body>\n"
            "</html>"
        ),
        "language": "html",
        "title": "Demo Canvas",
    }
