"""A group-stage hybrid plays one player's settings as white and another's as black."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evolve_tonkinho import side_genes  # noqa: E402


def test_hybrid_uses_the_settings_of_its_colour():
    hybrid = {"white": {"temperature": 0.14}, "black": {"temperature": 0.02}}
    assert side_genes(hybrid, "white") == {"temperature": 0.14}
    assert side_genes(hybrid, "black") == {"temperature": 0.02}


def test_plain_players_are_the_same_on_both_colours():
    plain = {"temperature": 0.05}
    assert side_genes(plain, "white") is plain
    assert side_genes(plain, "black") is plain
