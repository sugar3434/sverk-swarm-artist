"""
Тест устойчивости LLM-клиента к сбою связи (обработка ошибок интеграции).

Моделирует сбой: создаёт OpenAICompatibleClient с заведомо неверным
SVERK_LLM_BASE_URL (адрес, на котором никто не слушает), оборачивает его в
FallbackLLMClient с OfflineRuleBasedClient() как fallback, вызывает chat(...)
и убеждается, что:
    1. Вызов НЕ падает никаким исключением.
    2. Возвращённая строка непустая (значит fallback реально сработал и
       выдал реплику офлайн-клиента).

Это прямая проверка судейского критерия «надёжность, обработка ошибок»:
сбой сети во время соревнования не должен прерывать публичный диалог агентов.

Запуск: `python3 tests/test_llm_client_fallback.py`
Совместим с pytest (использует обычные функции test_* и assert).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.llm_client import (  # noqa: E402
    FallbackLLMClient,
    LLMUnavailableError,
    OfflineRuleBasedClient,
    OpenAICompatibleClient,
    build_llm_client,
)
from agents.personas import PERSONAS  # noqa: E402


def test_fallback_client_survives_unreachable_endpoint() -> None:
    # Порт 1 на loopback — зарезервированный порт, на котором гарантированно
    # никто не слушает, поэтому соединение будет быстро отклонено ОС.
    primary = OpenAICompatibleClient(
        base_url="http://127.0.0.1:1",
        api_key="заведомо-неверный-ключ-для-теста",
        model="test-model",
        timeout_s=3.0,
    )
    fallback = OfflineRuleBasedClient(seed=5)
    client = FallbackLLMClient(primary=primary, fallback=fallback)

    system_prompt = PERSONAS["Академист"]["system_prompt"]
    user_prompt = "Промпт: нарисуй сокола. Текущий план (JSON): []"

    # Не должно бросить исключение — весь смысл FallbackLLMClient в этом.
    result = client.chat(system_prompt, user_prompt)

    assert isinstance(result, str), "chat() должен вернуть строку"
    assert result.strip() != "", "Fallback должен вернуть непустую реплику офлайн-клиента"
    print(f"OK: FallbackLLMClient пережил недоступный эндпоинт, ответ: {result!r}")


def test_primary_alone_raises_llm_unavailable_error() -> None:
    """Без обёртки FallbackLLMClient сбой должен явно проявляться как
    LLMUnavailableError (контролируемое исключение), а не как сетевой сбой
    произвольного типа или зависание."""
    primary = OpenAICompatibleClient(
        base_url="http://127.0.0.1:1",
        api_key="заведомо-неверный-ключ-для-теста",
        model="test-model",
        timeout_s=3.0,
    )
    try:
        primary.chat("системный промпт", "пользовательский промпт")
    except LLMUnavailableError:
        print("OK: OpenAICompatibleClient корректно бросает LLMUnavailableError при недоступном эндпоинте.")
    else:
        raise AssertionError("Ожидалось LLMUnavailableError при недоступном эндпоинте")


def test_build_llm_client_without_api_key_returns_offline() -> None:
    """Если SVERK_LLM_API_KEY не задан — фабрика должна сразу вернуть
    OfflineRuleBasedClient без каких-либо сетевых попыток."""
    old_value = os.environ.pop("SVERK_LLM_API_KEY", None)
    try:
        client = build_llm_client()
        assert isinstance(client, OfflineRuleBasedClient), (
            "Без API-ключа фабрика должна вернуть OfflineRuleBasedClient"
        )
        print("OK: build_llm_client() без ключа возвращает OfflineRuleBasedClient.")
    finally:
        if old_value is not None:
            os.environ["SVERK_LLM_API_KEY"] = old_value


def test_build_llm_client_with_api_key_wraps_in_fallback() -> None:
    """Если SVERK_LLM_API_KEY задан — фабрика должна вернуть FallbackLLMClient,
    оборачивающий основной клиент офлайн-заглушкой."""
    old_value = os.environ.get("SVERK_LLM_API_KEY")
    old_base_url = os.environ.get("SVERK_LLM_BASE_URL")
    os.environ["SVERK_LLM_API_KEY"] = "тестовый-ключ"
    os.environ["SVERK_LLM_BASE_URL"] = "http://127.0.0.1:1"
    try:
        client = build_llm_client()
        assert isinstance(client, FallbackLLMClient), (
            "С заданным API-ключом фабрика должна вернуть FallbackLLMClient"
        )
        # Убеждаемся, что даже через фабрику итоговый клиент устойчив к сбою сети.
        result = client.chat(PERSONAS["Минималист"]["system_prompt"], "Промпт: тест. Текущий план (JSON): []")
        assert result.strip() != ""
        print("OK: build_llm_client() с ключом возвращает FallbackLLMClient и переживает сбой сети.")
    finally:
        if old_value is not None:
            os.environ["SVERK_LLM_API_KEY"] = old_value
        else:
            os.environ.pop("SVERK_LLM_API_KEY", None)
        if old_base_url is not None:
            os.environ["SVERK_LLM_BASE_URL"] = old_base_url
        else:
            os.environ.pop("SVERK_LLM_BASE_URL", None)


def _run_all() -> None:
    test_fallback_client_survives_unreachable_endpoint()
    test_primary_alone_raises_llm_unavailable_error()
    test_build_llm_client_without_api_key_returns_offline()
    test_build_llm_client_with_api_key_wraps_in_fallback()
    print("\nВСЕ ТЕСТЫ test_llm_client_fallback.py ПРОЙДЕНЫ.")


if __name__ == "__main__":
    _run_all()
