# Контракт модулей — «Рой дронов-художников» (проект sverk-swarm-artist)

Этот файл — единый источник истины по интерфейсам между модулями, чтобы
независимо разрабатываемые части кода стыковались без переделок.
Язык кода и docstring — русский (как в регламенте и репозитории sverk-ros2).
Python 3.10+, без внешних тяжёлых зависимостей кроме: `numpy`, `Pillow`,
`pyyaml`, `requests`, `fastapi`+`uvicorn` (только в openclaw/drone_agent_server.py),
`rclpy`+`sverk_interfaces` (только в openclaw/drone_link.py, LocalDroneLink).
Модули vision/, agents/, swarm/conflict_scheduler.py, swarm/canvas_grid.py
НЕ должны импортировать rclpy — они обязаны работать и тестироваться офлайн,
без дрона и без ROS.

## Проверенный API sverk_interfaces (реальный, из клонированного репозитория)

```python
import sverk_interfaces
drone = sverk_interfaces.init(
    Nodename="имя_ноды",
    offboard_namespace="",       # префикс топиков/сервисов offboard_control (пусто = /navigate, /land, ...)
    fcu_namespace="/fmu_control",
    servo_enable="/servo_control/enable",
    servo_angle_topic="/servo_control/target_angle_deg",
    servo_center="/servo_control/center",
)

# Полёт (offboard_interfaces, drone.control):
drone.control.navigate(x=, y=, z=, yaw=, speed=, frame_id="body"|"map", auto_arm=True, timeout=)   # неблокирующий старт манёвра
drone.control.navigate_wait(x=, y=, z=, yaw=, speed=, frame_id=, auto_arm=, timeout=, tolerance=)  # блокирует до прибытия, кидает TimeoutError/RuntimeError
drone.control.land(timeout=)
drone.control.get_telemetry(frame_id="map")             # .x .y .z .yaw
drone.control.get_status(timeout=)                        # .battery_pct .armed .offboard_active .active_maneuver .last_abort_reason (None при таймауте)
drone.control.emergency_stop(land=True/False)              # мгновенная остановка манёвра
drone.control.return_to_launch()

# FMU / безопасность (drone.fcu):
drone.fcu.arm() / .disarm() / .force_disarm() / .kill_switch() / .reboot()   # Trigger.Response(.success, .message)

# Сервопривод клапана краски (drone.gpio), у нас 1 канал на дрон (своя форсунка):
drone.gpio.servo_enable() / .servo_disable()
drone.gpio.servo_set_angle(degrees: float)   # 0..180, публикация без ответа
drone.gpio.servo_center()
drone.gpio.servo_select_channel(n)           # если на дроне вдруг >1 форсунки — не используется в базовой версии

drone.close()   # обязательно в finally
```

Свободный сервис / топик по имени (для расширений): `drone.service.call(name, srv_type, **kwargs)`,
`drone.topic.subscribe/create_publisher/wait_for_message`.

Важно: у offboard_control уже ЕСТЬ аппаратный RC-перехват (kill switch с пульта,
параметр `check_kill_switch: true`, включён по умолчанию) — регламентный п. 2.6.11
"KILL SWITCH" в первую очередь обеспечивается ЭТИМ штатным механизмом фреймворка.
Наш код (openclaw/safety.py) добавляет ПРОГРАММНЫЙ дублирующий канал
(`drone.fcu.kill_switch()` / `drone.control.emergency_stop(land=True)`), но не должен
его подменять или отключать.

## Общие структуры данных (единые для всех модулей)

Файл `common/schema.py` (создаётся первым, все остальные модули импортируют из него):

```python
from dataclasses import dataclass, field
from typing import Optional

COLORS = ("black", "red", "blue", "yellow")  # чёрный=Академист, красный=Экспрессионист,
                                              # синий=Минималист, жёлтый/белый=Детализатор

@dataclass
class PaintTask:
    cell: str                 # id ячейки сетки полотна, напр. "B3" (буква=столбец, цифра=строка)
    color: str                # один из COLORS
    duration_s: float = 1.5   # длительность распыления, с
    passes: int = 1           # число проходов (Детализатор просит больше)
    priority: int = 0         # чем меньше — тем раньше (после диалога агентов)
    x: Optional[float] = None # мировые координаты (м), заполняет canvas_grid
    y: Optional[float] = None
    z: Optional[float] = None
    note: str = ""            # почему агент так решил (для лога/трансляции)

@dataclass
class DialogueTurn:
    agent: str
    text: str
    ts: float

@dataclass
class Plan:
    prompt: str
    cells: list             # list[PaintTask]
    transcript: list        # list[DialogueTurn]
    outline_color: str = "black"
    notes: str = ""

@dataclass
class ScheduledTask:
    task: PaintTask
    drone_id: str
    start_offset_s: float
    end_offset_s: float
```

## Разбиение работы

1. **vision/** (без ROS): `prompt_to_bitmap.py` (промпт → растр NxM), `bitmap_to_plan.py`
   (растр → список `PaintTask` без x/y/приоритета — их заполнят canvas_grid и диалог).
2. **agents/** (без ROS): `personas.py`, `llm_client.py`, `dialogue_engine.py`, `broadcast.py`.
   `run_dialogue(prompt, draft_tasks, config) -> Plan`.
3. **swarm/canvas_grid.py** и **swarm/conflict_scheduler.py** (без ROS):
   `CanvasGrid.cell_to_world(cell_id) -> (x,y,z)`, `schedule(tasks, fleet_cfg) -> list[ScheduledTask]`.
4. **openclaw/** и **swarm/fleet_coordinator.py** (использует rclpy/sverk_interfaces) —
   пишет ведущий агент (интегратор), т.к. требует точного соответствия API выше.
5. **mission_runner.py**, **README.md**, **sverk_code.md** — интегратор.

Все модули обязаны иметь `if __name__ == "__main__":` demo или `tests/test_*.py`,
работающие БЕЗ дрона и без сети (используя офлайн-заглушки).
