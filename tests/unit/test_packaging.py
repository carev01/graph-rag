"""Every package under src/ must be in the wheel -- the image installs the wheel
(non-editable), so a missing entry silently removes a whole service from it.
`theme_builder` was missing until 2026-09-24."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_every_src_package_is_in_the_wheel():
    listed = set(tomllib.loads((ROOT / "pyproject.toml").read_text())
                 ["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])
    present = {f"src/{p.parent.name}" for p in (ROOT / "src").glob("*/__init__.py")}
    assert present - listed == set(), f"missing from the wheel: {sorted(present - listed)}"


def test_dockerignore_keeps_secrets_and_data_out():
    ignored = (ROOT / ".dockerignore").read_text().split()
    for entry in (".env", ".env.*", "data", ".venv", ".git", ".superpowers"):
        assert entry in ignored, f".dockerignore must exclude {entry}"
