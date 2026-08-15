import os
import sys
import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    from src.tracing import init_phoenix
    phoenix_ok = init_phoenix(project_name="qa-review")
    if not phoenix_ok:
        logger.warning("Phoenix tracing not available")

    from src.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()
