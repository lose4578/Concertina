import json
import re
import tomllib
from importlib import metadata
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared_lerobot_commit() -> str:
    dependencies = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    specs = [d for d in dependencies if re.match(r"lerobot\s*[\[@]", d)]
    assert len(specs) == 1, f"Expected exactly one lerobot requirement in {PYPROJECT}, found {specs}"
    match = re.search(r"@\s*git\+\S+@([0-9a-f]{40})\s*$", specs[0])
    assert match, f"The lerobot requirement is not pinned to a full commit hash: {specs[0]!r}"
    return match.group(1)


def test_installed_lerobot_is_the_declared_pin():
    declared = _declared_lerobot_commit()
    direct_url = json.loads(metadata.distribution("lerobot").read_text("direct_url.json") or "{}")
    installed = direct_url.get("vcs_info", {}).get("commit_id")
    assert installed == declared, (
        f"Installed lerobot ({installed or direct_url.get('url', 'unknown source')}) is not the commit "
        f"declared in pyproject.toml ({declared}). Every other contract test records behavior of the "
        "declared commit; reinstall with `pip install -e .` or, if upgrading on purpose, re-audit the "
        "contracts in tests/contract/ against the new upstream source and update them together with "
        "the pin."
    )
