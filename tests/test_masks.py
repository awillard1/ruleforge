"""Tests for ruleforge/masks.py"""

import pytest
from ruleforge.masks import MaskLearner, password_to_mask, _char_class


class TestCharClass:
    def test_uppercase(self):
        assert _char_class("A") == "?u"

    def test_lowercase(self):
        assert _char_class("a") == "?l"

    def test_digit(self):
        assert _char_class("5") == "?d"

    def test_special(self):
        assert _char_class("!") == "?s"


class TestPasswordToMask:
    def test_simple(self):
        mask = password_to_mask("abc")
        assert mask == "?l?l?l"

    def test_mixed(self):
        mask = password_to_mask("Ab1!")
        assert mask == "?u?l?d?s"

    def test_football2025(self):
        mask = password_to_mask("Football2025!")
        assert mask.startswith("?u?l?l?l?l?l?l?l")
        assert "?d" in mask
        assert mask.endswith("?s")


class TestMaskLearner:
    @pytest.fixture
    def learner(self):
        return MaskLearner(min_length=4)

    def test_learn_builds_stats(self, learner):
        passwords = ["Football2025!", "password", "ABC123!", "hello"]
        learner.learn(passwords)
        assert len(learner._mask_stats) > 0

    def test_ranked_non_empty(self, learner):
        learner.learn(["password", "Password1!", "abc123"])
        ranked = learner.ranked()
        assert len(ranked) > 0

    def test_ranked_descending_score(self, learner):
        learner.learn(["password"] * 5 + ["Password1!"] * 2 + ["abc123"])
        ranked = learner.ranked()
        scores = [m.score for m in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_n(self, learner):
        learner.learn(["password", "Password1!", "abc123", "Test123!"])
        top = learner.ranked(top_n=2)
        assert len(top) <= 2

    def test_export_hcmask(self, learner, tmp_path):
        learner.learn(["Football2025!", "password", "Abc123!"])
        path = tmp_path / "masks.hcmask"
        learner.export_hcmask(path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) > 0

    def test_export_json(self, learner, tmp_path):
        learner.learn(["Football2025!"])
        path = tmp_path / "masks.json"
        learner.export_json(path)
        assert path.exists()

    def test_clusters_by_length(self, learner):
        learner.learn(["pass", "passwd1", "Password1!"])
        clusters = learner.clusters_by_length()
        assert len(clusters) > 0

    def test_short_password_ignored(self, learner):
        learner.learn(["ab"])  # too short (min=4)
        assert learner._total_passwords == 0

    def test_stats(self, learner):
        learner.learn(["password", "Password1!"])
        s = learner.stats()
        assert s["total_passwords"] == 2
        assert s["unique_masks"] >= 1
