"""Resolution and validation of --bruker_sdk_path.

The Bruker timsdata SDK unpacks to a folder holding win64/ and linux64/, so the
library sits one level below what a user naturally points at. These cover the
resolution order, the bounded search, and the fail-fast behaviour that keeps a
specified-but-unresolvable SDK from silently degrading to the approximation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.errors import JModError
from src.utils.io.file_reader import resolve_bruker_sdk_path


@pytest.fixture
def fake_sdk(tmp_path):
    """A stand-in for the real SDK tree, including its thirdparty/ decoy."""
    root = tmp_path / "timsdata"
    for sub in ("win64", "linux64", "include/c", "examples", "thirdparty"):
        (root / sub).mkdir(parents=True)
    (root / "win64" / "timsdata.dll").write_bytes(b"")
    (root / "linux64" / "libtimsdata.so").write_bytes(b"")
    return root


def test_sdk_root_resolves_to_linux_library(fake_sdk):
    got = resolve_bruker_sdk_path(str(fake_sdk), platform="linux")
    assert got == str(fake_sdk / "linux64" / "libtimsdata.so")
    assert os.path.isabs(got) and os.path.isfile(got)


def test_sdk_root_resolves_to_windows_library(fake_sdk):
    got = resolve_bruker_sdk_path(str(fake_sdk), platform="win32")
    assert got == str(fake_sdk / "win64" / "timsdata.dll")


def test_platform_directory_directly(fake_sdk):
    got = resolve_bruker_sdk_path(str(fake_sdk / "linux64"), platform="linux")
    assert got == str(fake_sdk / "linux64" / "libtimsdata.so")


def test_library_file_directly(fake_sdk):
    lib = fake_sdk / "linux64" / "libtimsdata.so"
    assert resolve_bruker_sdk_path(str(lib), platform="linux") == str(lib)


def test_explicit_file_wins_regardless_of_name(tmp_path):
    # A versioned or renamed copy is taken at the user's word.
    lib = tmp_path / "libtimsdata.so.2.21"
    lib.write_bytes(b"")
    assert resolve_bruker_sdk_path(str(lib), platform="linux") == str(lib)


def test_found_two_levels_down(tmp_path):
    lib = tmp_path / "vendor" / "linux64" / "libtimsdata.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    assert resolve_bruker_sdk_path(str(tmp_path), platform="linux") == str(lib)


def test_not_found_three_levels_down(tmp_path):
    lib = tmp_path / "a" / "b" / "c" / "libtimsdata.so"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    with pytest.raises(JModError):
        resolve_bruker_sdk_path(str(tmp_path), platform="linux")


def test_search_is_deterministic(tmp_path):
    # Created out of alphabetical order: an unsorted iterdir would be free to
    # return either, and on some filesystems would return bbb first.
    for name in ("bbb", "aaa"):
        d = tmp_path / name
        d.mkdir()
        (d / "libtimsdata.so").write_bytes(b"")

    expected = str(tmp_path / "aaa" / "libtimsdata.so")
    for _ in range(5):
        assert resolve_bruker_sdk_path(str(tmp_path), platform="linux") == expected


def test_platform_directory_beats_general_search(fake_sdk):
    # aaa/ sorts before linux64/, so only the explicit platform-dir check keeps
    # the real SDK layout winning.
    decoy = fake_sdk / "aaa"
    decoy.mkdir()
    (decoy / "libtimsdata.so").write_bytes(b"")

    got = resolve_bruker_sdk_path(str(fake_sdk), platform="linux")
    assert got == str(fake_sdk / "linux64" / "libtimsdata.so")


def test_shallower_hit_beats_deeper(tmp_path):
    shallow = tmp_path / "zzz" / "libtimsdata.so"
    shallow.parent.mkdir(parents=True)
    shallow.write_bytes(b"")
    deep = tmp_path / "aaa" / "nested" / "libtimsdata.so"
    deep.parent.mkdir(parents=True)
    deep.write_bytes(b"")

    assert resolve_bruker_sdk_path(str(tmp_path), platform="linux") == str(shallow)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unspecified_returns_none(value):
    # The only path to the no-SDK approximation.
    assert resolve_bruker_sdk_path(value, platform="linux") is None


def test_quotes_are_stripped(fake_sdk):
    got = resolve_bruker_sdk_path(f'"{fake_sdk}"', platform="linux")
    assert got == str(fake_sdk / "linux64" / "libtimsdata.so")


def test_missing_path_raises(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(JModError) as exc:
        resolve_bruker_sdk_path(str(missing), platform="linux")

    assert str(missing) in str(exc.value)
    assert "settings.json" in str(exc.value)


def test_directory_without_library_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(JModError) as exc:
        resolve_bruker_sdk_path(str(tmp_path), platform="linux")
    assert "libtimsdata.so" in str(exc.value)


def test_wrong_platform_library_raises(fake_sdk):
    # linux64/libtimsdata.so is present, but a Windows run needs timsdata.dll.
    (fake_sdk / "win64" / "timsdata.dll").unlink()
    with pytest.raises(JModError) as exc:
        resolve_bruker_sdk_path(str(fake_sdk), platform="win32")
    assert "timsdata.dll" in str(exc.value)


def test_darwin_resolves_to_the_windows_library(fake_sdk):
    # Bruker publishes no macOS build; its Windows library runs under Wine.
    got = resolve_bruker_sdk_path(str(fake_sdk), platform="darwin")
    assert got == str((fake_sdk / "win64" / "timsdata.dll").resolve())


def test_darwin_unspecified_returns_none():
    assert resolve_bruker_sdk_path(None, platform="darwin") is None


@pytest.mark.parametrize("plat", ["linux", "win32", "darwin"])
def test_non_strict_returns_none_without_raising(tmp_path, plat):
    # The GUI validates candidate paths with strict=False: no exception.
    assert resolve_bruker_sdk_path(str(tmp_path / "nope"),
                                   platform=plat, strict=False) is None


def test_non_strict_still_resolves_a_good_path(fake_sdk):
    got = resolve_bruker_sdk_path(str(fake_sdk), platform="linux", strict=False)
    assert got == str(fake_sdk / "linux64" / "libtimsdata.so")


class TestResolveBrukerSetting:
    """CLI-vs-settings precedence and persistence in run_jmod."""

    @pytest.fixture
    def settings_file(self, tmp_path, monkeypatch):
        import src.utils.gui_utils as gui_utils

        path = tmp_path / "settings" / "settings.json"
        monkeypatch.setattr(gui_utils, "SETTINGS_DIR", path.parent)
        monkeypatch.setattr(gui_utils, "SETTINGS_FILE", path)
        return path

    @pytest.fixture(autouse=True)
    def _linux(self, monkeypatch):
        # Pin the platform so these run identically on a macOS dev box and in
        # the linux/amd64 container.
        import src.utils.io.file_reader as file_reader

        real = file_reader.resolve_bruker_sdk_path
        monkeypatch.setattr(
            file_reader,
            "resolve_bruker_sdk_path",
            lambda p, **kw: real(p, platform="linux"),
        )

    def _read(self, settings_file):
        import json

        return json.loads(settings_file.read_text())

    def test_cli_path_is_resolved_and_persisted(self, fake_sdk, settings_file):
        from src.utils.io.file_reader import resolve_bruker_setting

        got = resolve_bruker_setting(str(fake_sdk))

        expected = str(fake_sdk / "linux64" / "libtimsdata.so")
        assert got == expected
        # The resolved file is stored, not the directory the user typed.
        assert self._read(settings_file)["bruker_sdk_path"] == expected

    def test_bad_cli_path_does_not_touch_settings(self, tmp_path, settings_file):
        from src.utils.io.file_reader import resolve_bruker_setting

        with pytest.raises(JModError):
            resolve_bruker_setting(str(tmp_path / "nope"))

        # Regression: the old code persisted before validating, so a typo became
        # the stored default for every later run.
        assert not settings_file.exists()

    def test_stored_setting_used_when_no_cli_arg(self, fake_sdk, settings_file):
        import json

        from src.utils.io.file_reader import resolve_bruker_setting

        lib = str(fake_sdk / "linux64" / "libtimsdata.so")
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps({"bruker_sdk_path": lib}))

        assert resolve_bruker_setting(None) == lib

    def test_stored_setting_is_validated_too(self, tmp_path, settings_file):
        import json

        from src.utils.io.file_reader import resolve_bruker_setting

        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps({"bruker_sdk_path": str(tmp_path / "gone")}))

        # A stale stored path fails loudly rather than degrading to the
        # approximation, which would silently produce non-matching m/z.
        with pytest.raises(JModError):
            resolve_bruker_setting(None)

    def test_nothing_specified_returns_none(self, settings_file):
        from src.utils.io.file_reader import resolve_bruker_setting

        assert resolve_bruker_setting(None) is None

    def test_empty_cli_arg_clears_stored_setting(self, fake_sdk, settings_file):
        import json

        from src.utils.io.file_reader import resolve_bruker_setting

        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(
            json.dumps({"bruker_sdk_path": str(fake_sdk / "linux64" / "libtimsdata.so")})
        )

        assert resolve_bruker_setting("") is None
        assert self._read(settings_file)["bruker_sdk_path"] is None


# --- which command computes the calibration tables, per platform -----------

class _Done:
    returncode = 0
    stdout = "ok"
    stderr = ""


@pytest.fixture
def capture_run(monkeypatch):
    """Stub subprocess.run: record the command and write calibration.json."""
    import subprocess
    seen = {}

    def run(cmd, env=None, **kw):
        seen["cmd"], seen["env"] = cmd, env
        out = cmd[-1]
        if out.startswith("Z:"):        # Wine path back to the Mac path
            out = out[2:].replace("\\", "/")
        os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "calibration.json"), "w").write("{}")
        return _Done()
    monkeypatch.setattr(subprocess, "run", run)
    return seen


@pytest.mark.parametrize("plat", ["linux", "win32"])
def test_linux_and_windows_run_the_sdk_natively(tmp_path, monkeypatch, capture_run, plat):
    from src.utils.io import file_reader as fr
    monkeypatch.setattr(sys, "platform", plat)
    d = tmp_path / "run.d"; d.mkdir()
    tables = fr._sdk_calibration_tables(str(d), "/sdk/libtimsdata.so")
    cmd = capture_run["cmd"]
    assert cmd[0] == sys.executable and cmd[1] == fr._SDK_TABLES_SCRIPT
    assert cmd[2] == "/sdk/libtimsdata.so" and capture_run["env"] is None
    assert os.path.isfile(os.path.join(tables, "calibration.json"))


def test_macos_runs_the_windows_sdk_under_wine(tmp_path, monkeypatch, capture_run):
    from src.utils.io import file_reader as fr
    wine = tmp_path / "wine"; wine.write_text("")
    py = tmp_path / "python.exe"; py.write_text("")
    monkeypatch.setattr(fr, "_WINE", str(wine))
    monkeypatch.setattr(fr, "_WINE_PYTHON", str(py))
    monkeypatch.setattr(fr, "_WINE_PREFIX", str(tmp_path / "prefix"))
    monkeypatch.setattr(sys, "platform", "darwin")
    d = tmp_path / "run.d"; d.mkdir()
    fr._sdk_calibration_tables(str(d), str(tmp_path / "sdk" / "timsdata.dll"))
    cmd = capture_run["cmd"]
    assert cmd[:2] == [str(wine), str(py)]
    assert cmd[3] == fr._wine_path(str(tmp_path / "sdk" / "timsdata.dll"))
    assert capture_run["env"]["WINEPREFIX"] == str(tmp_path / "prefix")


def test_macos_without_wine_fails(tmp_path, monkeypatch, capture_run):
    from src.utils.io import file_reader as fr
    monkeypatch.setattr(fr, "_WINE", str(tmp_path / "no-wine"))
    monkeypatch.setattr(sys, "platform", "darwin")
    d = tmp_path / "run.d"; d.mkdir()
    with pytest.raises(FileNotFoundError):
        fr._sdk_calibration_tables(str(d), "/sdk/timsdata.dll")
    assert "cmd" not in capture_run


@pytest.mark.parametrize("plat", ["linux", "win32", "darwin"])
def test_the_sdk_is_required(tmp_path, monkeypatch, capture_run, plat):
    from src.utils.io import file_reader as fr
    monkeypatch.setattr(sys, "platform", plat)
    d = tmp_path / "run.d"; d.mkdir()
    with pytest.raises(JModError):
        fr._sdk_calibration_tables(str(d), None)
    assert "cmd" not in capture_run


def test_cached_tables_are_reused(tmp_path, capture_run):
    from src.utils.io import file_reader as fr
    d = tmp_path / "run.d"; (d / fr._SDK_TABLES_DIR).mkdir(parents=True)
    (d / fr._SDK_TABLES_DIR / "calibration.json").write_text("{}")
    fr._sdk_calibration_tables(str(d), "/sdk/libtimsdata.so")
    assert "cmd" not in capture_run
