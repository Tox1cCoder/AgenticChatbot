"""Composition of project-level and conversation-level instructions."""

from app.utils.text_processing import compose_system_instruction, sanitize_persona


def test_returns_none_when_both_absent():
    assert compose_system_instruction(None, None) is None
    assert compose_system_instruction("   ", "") is None


def test_persona_only_passes_through_unchanged():
    assert compose_system_instruction(None, "Be terse.") == "Be terse."


def test_project_only_passes_through_unchanged():
    result = compose_system_instruction(
        "Answer in Vietnamese.", None
    )
    assert result == "Answer in Vietnamese."


def test_both_present_are_headered_with_project_first():
    result = compose_system_instruction(
        "Answer in Vietnamese.", "Be terse."
    )
    assert result == (
        "Project instructions:\nAnswer in Vietnamese.\n\n"
        "Conversation-specific instructions:\nBe terse."
    )


def test_each_part_is_capped_independently_so_the_persona_survives():
    """The bug this guards: compose-then-truncate would drop the
    persona entirely, because the project text leads and the cap is
    8000."""
    result = compose_system_instruction("P" * 9000, "Q" * 9000)

    assert result.count("P") == 8001
    assert result.count("Q") == 8000
    assert result.endswith("Q" * 100)


def test_project_less_output_is_byte_identical_to_the_previous_behaviour():
    """Every conversation that predates projects must render exactly as
    before."""
    persona = "  Be   terse.\n\n\n\nAlways answer in full sentences.  "

    assert compose_system_instruction(None, persona) == (
        sanitize_persona(persona)
    )
