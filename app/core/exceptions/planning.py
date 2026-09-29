from enum import Enum


class PauseReason(str, Enum):
    RECURSION_LIMIT = "recursion_limit"
    RATE_LIMIT = "rate_limit"
    MAX_TASKS_REACHED = "max_tasks_reached"
