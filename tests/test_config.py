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


class TestSetup:
    @pytest.fixture
    def inputs(self, monkeypatch):
        """A valid set of inputs, restored after each test."""
        monkeypatch.setattr(config.args, "config_json", None)
        monkeypatch.setattr(config.args, "speclib", "lib.tsv")
        monkeypatch.setattr(config.args, "mzml", ["a.mzML"])
        monkeypatch.setattr(config.args, "tag", "None")
        monkeypatch.setattr(config.args, "plexDIA", False)

    def test_no_spectral_library_raises(self, inputs, monkeypatch):
        monkeypatch.setattr(config.args, "speclib", None)
        with pytest.raises(JModError, match="No spectral library given"):
            config.setup()

    def test_no_data_files_raises(self, inputs, monkeypatch):
        monkeypatch.setattr(config.args, "mzml", None)
        with pytest.raises(JModError, match="No data files given"):
            config.setup()

    def test_single_data_file_becomes_a_list(self, inputs, monkeypatch):
        monkeypatch.setattr(config.args, "mzml", "a.mzML")
        config.setup()
        assert config.args.mzml == ["a.mzML"]

    def test_tag_turns_on_plexDIA(self, inputs, monkeypatch):
        monkeypatch.setattr(config.args, "tag", "mTRAQ")
        config.setup()
        assert config.args.plexDIA is True


class TestCommandLineOverrides:
    def test_only_typed_options_are_collected(self):
        given = vars(config._given_parser.parse_args(["--no_ms1_req", "-i", "a.mzML"]))
        assert given == {"no_ms1_req": False, "mzml": ["a.mzML"]}  # no defaults for the rest

    def test_typed_option_beats_the_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config.args, "ppm", config.args.ppm)
        monkeypatch.setattr(config, "cli_args", {"ppm": 5.0})
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ppm": 20.0}))
        config.load_config_from_json(str(path))
        assert config.args.ppm == 5.0

    def test_i_replaces_the_json_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config.args, "mzml", None)
        monkeypatch.setattr(config, "cli_args", {"mzml": ["c.mzML"]})
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"mzml": ["a.mzML", "b.mzML"]}))
        config.load_config_from_json(str(path))
        assert config.args.mzml == ["c.mzML"]
