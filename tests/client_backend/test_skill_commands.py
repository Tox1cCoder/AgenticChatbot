from types import SimpleNamespace

from shared.skills import commands


def test_windows_reparse_point_detection_does_not_require_path_is_junction(tmp_path, monkeypatch):
    reparse_flag = 0x400
    monkeypatch.setattr(
        commands.os,
        "lstat",
        lambda _path: SimpleNamespace(st_file_attributes=reparse_flag),
    )

    assert commands._has_windows_reparse_point(
        tmp_path / "junction",
        platform_name="nt",
    )
