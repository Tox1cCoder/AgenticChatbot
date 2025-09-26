"""
Utility functions for converting between camelCase and snake_case.
"""

import re
from typing import Dict, Any


def to_camel_case(snake_str: str) -> str:
    if not snake_str:
        return snake_str

    parts = snake_str.split("_")
    return parts[0] + "".join(word.capitalize() for word in parts[1:])


def to_snake_case(camel_str: str) -> str:
    if not camel_str:
        return camel_str

    snake_str = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_str)
    snake_str = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_str)
    return snake_str.lower()


def convert_dict_keys_to_snake_case(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict):
        return {
            to_snake_case(key): convert_dict_keys_to_snake_case(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [convert_dict_keys_to_snake_case(item) for item in data]
    else:
        return data


def convert_dict_keys_to_camel_case(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict):
        return {
            to_camel_case(key): convert_dict_keys_to_camel_case(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [convert_dict_keys_to_camel_case(item) for item in data]
    else:
        return data


to_camel = to_camel_case
