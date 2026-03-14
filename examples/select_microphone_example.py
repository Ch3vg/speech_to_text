"""List available microphones and start recognition on the selected one."""

from speech_to_text import Engine, SpeechToText, list_devices

devices = list_devices()

if not devices:
    print("Не найдено аудиоустройств ввода.")
    raise SystemExit(1)

print("Доступные микрофоны:\n")
for dev in devices:
    default = " (по умолчанию)" if dev["is_default"] else ""
    print(f"  [{dev['index']}] {dev['name']}{default}")

print()
choice = input("Введите номер устройства (Enter = по умолчанию): ").strip()

device = int(choice) if choice else None

print("\nГоворите... (Ctrl+C для остановки)\n")

try:
    for result in SpeechToText(Engine.VOSK, device=device, language="ru"):
        prefix = ">>>" if result.type.value == "final" else "..."
        print(f"{prefix} {result.text}")
except KeyboardInterrupt:
    print("\nОстановлено.")
