import os
import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path so `agents` can be imported when tests run.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--question",
        action="store",
        default=None,
        help="Override agent question used in tests (or set AGENT_TEST_QUESTION env)",
    )


@pytest.fixture
def user_question(request):
    """
    Resolve the question in this priority:
    1) CLI: --question "..."
    2) ENV: AGENT_TEST_QUESTION
    3) Module-level DEFAULT_QUESTION (per test file)
    4) Global fallback sample.
    """
    cli_q = request.config.getoption("--question")
    env_q = os.getenv("AGENT_TEST_QUESTION")
    module_q = getattr(request.module, "DEFAULT_QUESTION", None)
    return cli_q or env_q or module_q or "What was the closing price of Microsoft on March 15, 2024?"


def log_step(label: str, message: str):
    """Simple, readable log helper for test steps."""
    print(f"[{label}] {message}")

