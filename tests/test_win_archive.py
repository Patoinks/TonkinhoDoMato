"""Every TonkAI win is archived (PGN + JSON for the site); nothing else is."""
import io
import json

import chess
import chess.pgn
import pytest

from tonkai import web_publish
from tonkai.win_history import WinArchive


def mate(moves):
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board


SCHOLAR = ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#")      # white wins
FOOL = ("f3", "e5", "g4", "Qh4#")                                 # black wins


def test_every_win_is_appended_and_indexed(tmp_path):
    archive = WinArchive(tmp_path / "wins")
    assert archive.add(mate(SCHOLAR), chess.WHITE, "Rodent IV", "1-0", "checkmate", game=12)
    assert archive.add(mate(FOOL), chess.BLACK, "Rodent IV", "0-1", "checkmate", game=40)
    wins = archive.load()
    assert [w["number"] for w in wins] == [1, 2]
    assert [w["plies"] for w in wins] == [7, 4]
    assert wins[1]["tonkai_color"] == "black" and wins[1]["game"] == 40
    text = (tmp_path / "wins/wins.pgn").read_text(encoding="utf-8")
    stream, games = io.StringIO(text), []
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    assert [g.headers["Result"] for g in games] == ["1-0", "0-1"]
    assert games[1].headers["Black"] == "TonkAI"
    assert json.loads((tmp_path / "wins/wins.json").read_text())["format"] == "tonkai-wins-v1"


def test_losses_draws_and_disabled_archive_write_nothing(tmp_path):
    archive = WinArchive(tmp_path / "wins")
    assert not archive.add(mate(SCHOLAR), chess.BLACK, "Rodent IV", "1-0", "checkmate")   # TonkAI lost
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert not archive.add(stalemate, chess.WHITE, "Rodent IV", "1/2-1/2", "stalemate")
    assert not (tmp_path / "wins").exists()
    assert not WinArchive("").add(mate(SCHOLAR), chess.WHITE, "Rodent IV", "1-0", "checkmate")


def test_contradictory_results_are_rejected(tmp_path):
    archive = WinArchive(tmp_path / "wins")
    with pytest.raises(ValueError):
        archive.add(mate(FOOL), chess.WHITE, "Rodent IV", "1-0", "checkmate")     # board says 0-1
    with pytest.raises(ValueError):
        archive.add(mate(SCHOLAR), chess.WHITE, "TonkAI", "1-0", "checkmate")     # needs an opponent
    assert archive.load() == []


def test_only_the_real_archive_is_ever_published(tmp_path, monkeypatch):
    config = tmp_path / "web.json"
    config.write_text(json.dumps({"enabled": True}))
    launched = []
    monkeypatch.setattr(web_publish.subprocess, "Popen", lambda *a, **k: launched.append(a) or "process")
    assert web_publish.publish_wins(tmp_path / "wins", config_path=config) is None
    assert web_publish.publish_wins(web_publish.ROOT / "history/wins", config_path=tmp_path / "none.json") is None
    assert web_publish.publish_wins(web_publish.ROOT / "history/wins", config_path=config) == "process"
    config.write_text(json.dumps({"enabled": False}))
    assert web_publish.publish_wins(web_publish.ROOT / "history/wins", config_path=config) is None
    assert len(launched) == 1


def test_featured_game_only_wins_rank_rating_length_then_recency(tmp_path):
    from tonkai.win_history import FeaturedGameRecorder
    featured = FeaturedGameRecorder(tmp_path / "featured_game.pgn")
    assert not featured.consider(mate(FOOL), chess.WHITE, "Stockfish", 3000, "0-1", "checkmate")
    assert not featured.consider(chess.Board(), chess.WHITE, "Stockfish", 3000, "1/2-1/2", "agreement")
    assert not featured.path.exists()
    def save(moves, color, elo, day):
        return featured.consider(mate(moves), color, "External", elo,
                                 "1-0" if color else "0-1", "checkmate",
                                 recorded_utc=f"2026-09-{day:02d}T12:00:00+00:00")
    assert save(SCHOLAR, chess.WHITE, 1000, 20)
    assert save(FOOL, chess.BLACK, 1500, 19)  # rating beats length and recency
    assert not save(SCHOLAR, chess.WHITE, 1000, 23)
    assert save(SCHOLAR, chess.WHITE, 1500, 18)  # length beats recency
    assert not save(FOOL, chess.BLACK, 1500, 23)
    assert save(SCHOLAR, chess.WHITE, 1500, 23)
    assert not save(SCHOLAR, chess.WHITE, 1500, 22)
    game = chess.pgn.read_game(io.StringIO(featured.path.read_text(encoding="utf-8")))
    assert game.headers["RecordedUTC"] == "2026-09-23T12:00:00+00:00"
    assert game.headers["Result"] == "1-0"


def test_featured_win_replaces_legacy_loss(tmp_path):
    from tonkai.win_history import FeaturedGameRecorder, build_win_game
    featured = FeaturedGameRecorder(tmp_path / "featured.pgn")
    game, _ = build_win_game(mate(FOOL), chess.WHITE, "Strong", "0-1", "checkmate")
    game.headers["OpponentElo"] = "3000"
    featured.path.write_text(str(game), encoding="utf-8")
    assert featured.stored_key() is None
    assert featured.consider(mate(SCHOLAR), chess.WHITE, "Weak", 800, "1-0", "checkmate")


def test_archive_selection_uses_real_games_ratings_and_timestamps(tmp_path):
    from tonkai.win_history import featured_win_pgn
    archive = WinArchive(tmp_path)
    archive.add(mate(SCHOLAR), chess.WHITE, "TinyOpponent depth 2", "1-0", "checkmate")
    archive.add(mate(FOOL), chess.BLACK, "Stockfish 11; nominal Elo 1500; 0.002s/move", "0-1", "checkmate")
    wins = archive.load()
    for w in wins: w["recorded_utc"] = "2026-09-20T12:00:00+00:00"
    # Higher rating wins, even with fewer moves. Do not rank by engine version.
    best = chess.pgn.read_game(io.StringIO(featured_win_pgn(wins)))
    assert best.headers["OpponentElo"] == "1500"
    assert best.headers["PlyCount"] == "4"
    longer = dict(wins[0], opponent_elo=1500, recorded_utc="2026-09-19T12:00:00+00:00")
    newest = dict(longer, recorded_utc="2026-09-23T12:00:00+00:00")
    # Deliberately shuffled; dates, not input order, break length/rating ties.
    best = chess.pgn.read_game(io.StringIO(featured_win_pgn([newest, *wins, longer])))
    assert best.headers["PlyCount"] == "7"
    assert best.headers["RecordedUTC"] == newest["recorded_utc"]
    loss = dict(wins[1], pgn=wins[1]["pgn"].replace('[Black "TonkAI"]', '[Black "Other"]')
                .replace('[White "Stockfish 11; nominal Elo 1500; 0.002s/move"]', '[White "TonkAI"]'),
                opponent_elo=3000)
    assert featured_win_pgn([loss, {"pgn": "bad"}]) == ""
    assert featured_win_pgn([]) == ""


def test_publisher_ignores_legacy_featured_loss(tmp_path, monkeypatch):
    from scripts import publish_wins as publisher
    wins = tmp_path / "wins"
    WinArchive(wins).add(mate(SCHOLAR), chess.WHITE, "Engine; nominal Elo 1500", "1-0", "checkmate")
    (wins / "featured_game.pgn").write_text("legacy loss must never be copied", encoding="utf-8")
    site = tmp_path / "site"
    for sub in ("public/tonkai", "dist/tonkai"):
        (site / sub).mkdir(parents=True)
    (site / "dist/index.html").write_bytes(b"same live index")
    cfg = tmp_path / "web.json"
    cfg.write_text(json.dumps({"enabled": True, "site_directory": str(site), "public_url": "https://example.test"}))
    monkeypatch.setattr(publisher, "WINS", wins)
    monkeypatch.setattr(publisher, "LONGEST", tmp_path / "absent.pgn")
    monkeypatch.setattr(publisher, "RECEIPT", tmp_path / "receipts/latest.json")
    monkeypatch.setattr(publisher.subprocess, "run", lambda *a, **kw: type("Result", (), {"returncode": 0})())
    def fetch(url):
        if url.endswith("/"): return b"same live index"
        return (site / "dist/tonkai" / url.split("/")[-1].split("?")[0]).read_bytes()
    monkeypatch.setattr(publisher, "fetch", fetch)
    assert publisher.publish(cfg) == "published 1 wins + featured game"
    exported = (site / "public/tonkai/featured-game.pgn").read_text(encoding="utf-8")
    game = chess.pgn.read_game(io.StringIO(exported))
    assert game.headers["Result"] == "1-0"
    assert game.headers["OpponentElo"] == "1500"


def test_publisher_index_guard_allows_only_whitespace_changes():
    from scripts.publish_wins import same_index

    live = b'<html>\n  <script src="/assets/index-abc.js"></script>\n</html>\n'
    local = b'<html> <script src="/assets/index-abc.js"></script> </html>'
    changed = b'<html> <script src="/assets/index-def.js"></script> </html>'
    assert same_index(live, local)
    assert not same_index(live, changed)
