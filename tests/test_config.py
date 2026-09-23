import json

import pytest

import src.config as config
from src.utils.errors import JModError


class TestDataFileFlag:
    def test_repeated_i_gives_a_list(self):
        args = config.parser.parse_args(["-i", "a.mzML", "-i", "b.mzML"])
        assert args.mzml == ["a.mzML", "b.mzML"]

    def test_no_i_gives_none(self):
        assert config.parser.parse_args([]).mzml is None


class TestLoadConfigFromJson:
    def test_mzml_list_is_loaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config.args, "mzml", None)
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"mzml": ["a.mzML", "b.mzML"]}))
        config.load_config_from_json(str(path))
        assert config.args.mzml == ["a.mzML", "b.mzML"]

    def test_unescaped_backslash_raises_with_hint(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"mzml": "C:\\Users\\me\\run.mzML"}')  # single backslashes: invalid JSON
        with pytest.raises(JModError, match="forward slashes"):
            config.load_config_from_json(str(path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(JModError, match="Could not open config JSON"):
            config.load_config_from_json(str(tmp_path / "missing.json"))
