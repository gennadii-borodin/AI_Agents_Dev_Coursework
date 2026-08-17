import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    from src.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
