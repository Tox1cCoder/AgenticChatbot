import pytest

from app.ai.tool_execution import execute_tool_calls
from app.ai.utils import normalize_tool_call


class _DummyTool:
    name = "tool_search"

    def __init__(self):
        self.received_args = None

    async def ainvoke(self, tool_args):
        self.received_args = tool_args
        return {"ok": True}


def test_normalize_tool_call_decodes_stringified_json_args():
    normalized = normalize_tool_call(
        {
            "name": "tool_search",
            "args": '{"query":"gi\\u00e1 x\\u0103ng d\\u1ea7u h\\u00f4m nay Petrolimex"}',
            "id": "call-1",
        }
    )

    assert normalized["args"] == {"query": "giá xăng dầu hôm nay Petrolimex"}


@pytest.mark.asyncio
async def test_execute_tool_calls_passes_decoded_vietnamese_args_to_tool():
    tool = _DummyTool()

    outputs, artifacts, images = await execute_tool_calls(
        tool_calls=[
            {
                "name": "tool_search",
                "args": '{"query":"gi\\u00e1 x\\u0103ng d\\u1ea7u h\\u00f4m nay Petrolimex"}',
                "id": "call-1",
            }
        ],
        tool_map={"tool_search": tool},
    )

    assert tool.received_args == {"query": "giá xăng dầu hôm nay Petrolimex"}
    assert outputs[0]["name"] == "tool_search"
    assert artifacts[0]["args"] == {"query": "giá xăng dầu hôm nay Petrolimex"}
    assert images == []
