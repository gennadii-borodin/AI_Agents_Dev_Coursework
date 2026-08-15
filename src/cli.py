import logging
import sys

import click
import yaml

from src.report import run_review
from src.tracing import init_phoenix

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def _print_config():
    from src.config import get_settings

    settings = get_settings().model_dump()
    for key in ("router_ai_api_key", "database_url"):
        if settings.get(key):
            settings[key] = "***"
    click.echo(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True))


def _print_help():
    print("\nДоступные сценарии:")
    print("  полное ревью            - анализ всех требований и тестов")
    print("  покрытие REQ-XXX        - проверка покрытия конкретного требования")
    print("  оценить дизайн          - оценка качества тест-дизайна")
    print("  стандарты               - проверка соответствия стандартам QA")
    print("  найти тесты без требований - поиск тестов без привязки к REQ\n")


@click.command()
@click.argument("query", required=False, default=None)
@click.option("--interactive", is_flag=True, help="Запуск в интерактивном режиме")
@click.option("--show-config", is_flag=True, help="Вывести эффективные настройки (без секретов)")
def main(query: str, interactive: bool, show_config: bool):
    """QA Review Agent - автоматизированное ревью тест-кейсов.

    Примеры использования:

        qa-review "провести полное ревью"
        qa-review "проверить покрытие REQ-001"
        qa-review "оценить дизайн тестов"
        qa-review "найти тесты без требований"
        qa-review --interactive
        qa-review --show-config
    """
    if show_config:
        _print_config()
        return

    init_phoenix(project_name="qa-review")

    if interactive:
        print("QA Review Agent - интерактивный режим")
        print("Доступные команды: help, exit, quit\n")
        while True:
            try:
                user_input = click.prompt("qa-review>", default="").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nВыход из интерактивного режима")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Выход")
                break
            if user_input.lower() == "help":
                _print_help()
                continue

            try:
                run_review(user_input)
            except Exception as e:
                logger.exception("Review failed")
                print(f"Ошибка: {e}")

    elif query:
        try:
            run_review(query)
        except Exception as e:
            logger.exception("Review failed")
            print(f"Ошибка: {e}")
            sys.exit(1)
    else:
        _print_help()


if __name__ == "__main__":
    main()
