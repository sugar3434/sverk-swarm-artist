"""
OpenCLaw — программный слой-посредник между решениями мультиагентной
LLM-системы и физическими исполнительными механизмами дронов «Сверх».

Реализован поверх высокоуровневого API `sverk_interfaces` (см.
https://github.com/sverk-tech/sverk-ros2). Архитектурно повторяет
уже существующий в фреймворке мост `web/ros2_mcp` (HTTP/MCP-сервер с
policy-движком для LLM-агентов) — см. docs/ARCHITECTURE_CONTRACT.md и
sverk_code.md, раздел «Почему OpenCLaw устроен именно так».
"""
