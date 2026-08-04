from src.llm import parse_tool_arguments


def test_parse_tool_arguments_from_json_string() -> None:
    parsed = parse_tool_arguments('{"latitude": 6.9, "longitude": 79.8}')
    assert parsed["latitude"] == 6.9
