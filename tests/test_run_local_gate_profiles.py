from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("run_local_gate", ROOT / "scripts/run_local_gate.py")
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_profile_suffixes() -> None:
    assert MODULE.profile_mode_suffix({"provider": "openai_compatible"}) == "nothink"
    assert MODULE.profile_mode_suffix(
        {"provider": "ollama_native", "structured_output": True}
    ) == "native_schema"
    assert MODULE.profile_mode_suffix(
        {
            "provider": "ollama_native",
            "structured_output": False,
            "think": False,
        }
    ) == "native_nothink_plain"
