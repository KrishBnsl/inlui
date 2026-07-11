import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "verify_gemini.py"


def _module():
    spec = importlib.util.spec_from_file_location("verify_gemini_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_live_script_is_opt_in(monkeypatch, capsys):
    module = _module()
    monkeypatch.delenv("RUN_LIVE_GEMINI_TEST", raising=False)
    assert module.main() == 0
    assert capsys.readouterr().out.strip() == (
        "LIVE GEMINI TEST: NOT RUN (set RUN_LIVE_GEMINI_TEST=1)"
    )


def test_opted_in_script_requires_key_and_never_falls_back(monkeypatch, capsys):
    module = _module()
    monkeypatch.setenv("RUN_LIVE_GEMINI_TEST", "1")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(module, "load_dotenv", lambda *args, **kwargs: False)
    assert module.main() == 2
    output = capsys.readouterr().out
    assert "LIVE GEMINI TEST: NOT RUN" in output
    assert "GOOGLE_API_KEY_missing" in output
    assert "SUCCESS" not in output


def test_retired_embedding_model_is_migrated_without_editing_env(monkeypatch):
    module = _module()
    monkeypatch.setenv("RAG_EMBED_MODEL", "models/text-embedding-004")
    configured = module.RETIRED_EMBED_MODELS.get(
        module.os.getenv("RAG_EMBED_MODEL"), module.DEFAULT_EMBED_MODEL
    )
    assert configured == "models/gemini-embedding-2"
