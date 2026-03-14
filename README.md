# speech-to-text

Python-библиотека непрерывного распознавания речи с подключаемыми движками.
Два стиля API (итератор / callback), три встроенных движка (Vosk, Whisper, Deepgram)
и возможность подключить собственный.

## Установка

Базовый пакет (без движков):

```bash
pip install git+https://github.com/Ch3vg/SpeechToText.git
```

С нужным движком:

```bash
pip install "speech-to-text[vosk] @ git+https://github.com/Ch3vg/SpeechToText.git"           # Vosk — минимальная задержка, CPU
pip install "speech-to-text[whisper] @ git+https://github.com/Ch3vg/SpeechToText.git"        # faster-whisper — CPU (auto-fallback)
pip install "speech-to-text[whisper-gpu] @ git+https://github.com/Ch3vg/SpeechToText.git"    # faster-whisper + CUDA (NVIDIA GPU)
pip install "speech-to-text[deepgram] @ git+https://github.com/Ch3vg/SpeechToText.git"       # Deepgram — облачный, WebSocket
pip install "speech-to-text[all] @ git+https://github.com/Ch3vg/SpeechToText.git"            # все движки сразу (без CUDA)
```

### Системные требования

- Python >= 3.10
- Микрофон (или виртуальное аудиоустройство)
- Для `whisper` — NVIDIA GPU рекомендуется (работает и на CPU, но медленнее)
- Для `deepgram` — интернет-соединение и API-ключ

## Быстрый старт

### Итератор — 2 строки

```python
from speech_to_text import SpeechToText, Engine

for result in SpeechToText(Engine.VOSK):
    print(result.text)
```

### Callback — 4 строки

```python
from speech_to_text import SpeechToText, Engine

stt = SpeechToText(Engine.VOSK, language="ru")
stt.on_result(lambda r: print(r.text))
stt.start()
input("Enter для остановки...")
stt.stop()
```

### Транскрипция файла — 1 строка

```python
from speech_to_text import SpeechToText, Engine

text = SpeechToText(Engine.WHISPER, source="recording.ogg").transcribe()
print(text)
```

`transcribe()` собирает все FINAL-результаты, склеивает в одну строку
и возвращает когда файл обработан.

### Context manager

```python
from speech_to_text import SpeechToText, Engine

with SpeechToText(Engine.VOSK) as stt:
    for result in stt:
        print(result.text)
```

## Встроенные движки

### Vosk

Локальный движок на базе Kaldi. Минимальная задержка (~0.3-0.5 с), работает на CPU.
Модели скачиваются автоматически при первом запуске.

```python
from speech_to_text import SpeechToText, Engine

stt = SpeechToText(Engine.VOSK, language="ru")

# Явное указание модели:
stt = SpeechToText(Engine.VOSK, model_name="vosk-model-small-ru-0.22")

# Локальная модель:
stt = SpeechToText(Engine.VOSK, model_path="/path/to/model")
```

| Параметр     | По умолчанию              | Описание                            |
|--------------|---------------------------|-------------------------------------|
| `language`   | `"ru"`                    | ISO-код языка для авто-выбора модели|
| `model_name` | —                         | Имя модели Vosk для авто-загрузки   |
| `model_path` | —                         | Путь к локальной распакованной модели|

### Whisper (faster-whisper)

Локальный движок на базе faster-whisper (CTranslate2). Высокая точность,
поддержка GPU. Буферизует аудио и транскрибирует при обнаружении паузы
(energy-based VAD). Поддерживает промежуточные результаты через
периодическую транскрипцию накопленного аудио.

При `compute_device="auto"` (по умолчанию) движок пытается использовать
CUDA; если библиотеки CUDA не установлены, автоматически переключается
на CPU с предупреждением в лог. Для явного использования GPU установите
пакет с CUDA-зависимостями: `pip install speech-to-text[whisper-gpu]`.

```python
from speech_to_text import SpeechToText, Engine, Device

stt = SpeechToText(Engine.WHISPER, model="small", language="ru")

# Тонкая настройка:
stt = SpeechToText(
    Engine.WHISPER,
    model="medium",
    language="ru",
    compute_device=Device.CUDA,  # Device.AUTO, Device.CPU, Device.CUDA
    compute_type="float16",      # "default", "float16", "int8"
    energy_threshold=300,        # порог энергии для детекции речи
    silence_duration=0.8,        # секунды тишины для завершения фразы
    partial_interval=2.0,        # интервал промежуточных результатов (сек)
)
```

| Параметр           | По умолчанию | Описание                                                  |
|--------------------|--------------|-----------------------------------------------------------|
| `model`            | `"small"`    | Размер модели: `tiny`, `base`, `small`, `medium`, `large` |
| `language`         | `"ru"`       | ISO-код языка                                             |
| `compute_device`   | `Device.AUTO`| Устройство: `Device.AUTO`, `Device.CPU`, `Device.CUDA`    |
| `compute_type`     | `"default"`  | Тип вычислений CTranslate2                                |
| `energy_threshold` | `300`        | RMS-порог для детекции речи                               |
| `silence_duration` | `0.8`        | Секунды тишины для финализации фразы                      |
| `partial_interval` | `2.0`        | Интервал между промежуточными транскрипциями               |

### Deepgram

Облачный движок через WebSocket. Минимальная задержка (~0.3-1 с),
высокая точность (Nova-3). Требует API-ключ ($200 бесплатных кредитов
при регистрации на [deepgram.com](https://deepgram.com)).

```python
from speech_to_text import SpeechToText, Engine

stt = SpeechToText(Engine.DEEPGRAM, api_key="YOUR_KEY", language="ru")
```

| Параметр       | По умолчанию  | Описание                              |
|----------------|---------------|---------------------------------------|
| `api_key`      | — (обязателен)| API-ключ Deepgram                     |
| `language`     | `"ru"`        | ISO-код языка                         |
| `model`        | `"nova-3"`    | Модель Deepgram                       |
| `smart_format` | `True`        | Авто-пунктуация и форматирование      |

## Источники звука

По умолчанию используется системный микрофон, но можно указать другой
источник через параметр `source`.

### Микрофон (по умолчанию)

```python
from speech_to_text import SpeechToText, Engine

# Системный микрофон по умолчанию
stt = SpeechToText(Engine.VOSK)

# Конкретный микрофон по индексу
stt = SpeechToText(Engine.VOSK, device=1)

# Конкретный микрофон по имени (подстрока, регистронезависимо)
stt = SpeechToText(Engine.VOSK, device="Realtek")
```

### Выбор микрофона

`list_devices()` возвращает список доступных устройств ввода:

```python
from speech_to_text import list_devices

for dev in list_devices():
    default = " *" if dev["is_default"] else ""
    print(f"  [{dev['index']}] {dev['name']}{default}")
```

Каждое устройство — dict с ключами `index`, `name`, `channels`,
`default_sample_rate`, `is_default`. Значение `index` передаётся
в параметр `device`.

### Аудиофайл

```python
from speech_to_text import SpeechToText, Engine, FileSource

# Короткий синтаксис — путь как строка
for result in SpeechToText(Engine.VOSK, source="recording.wav"):
    print(result.text)

# Явный FileSource с настройками
src = FileSource("recording.wav", realtime=False)
for result in SpeechToText(Engine.VOSK, source=src):
    print(result.text)
```

| Параметр   | По умолчанию | Описание                                                   |
|------------|--------------|------------------------------------------------------------|
| `path`     | — (обязателен)| Путь к аудиофайлу                                         |
| `realtime` | `True`       | Подавать чанки в реальном времени; `False` = максимальная скорость |
| `block_size`| `4000`      | Сэмплов на чанк (250 мс при 16 kHz)                       |

Форматы: WAV, OGG, FLAC, AIFF, MP3 и другие (все форматы `libsndfile`).
Любой sample rate, количество каналов и битность автоматически
конвертируются в 16 kHz mono int16.

### Байты (bytes)

Если аудио уже в памяти — передайте `bytes` напрямую в `source`:

```python
from speech_to_text import SpeechToText, Engine, BytesSource

# Короткий синтаксис — raw PCM bytes (16 kHz, mono, int16 LE)
pcm_data: bytes = get_audio_from_somewhere()
for result in SpeechToText(Engine.VOSK, source=pcm_data):
    print(result.text)

# Явный BytesSource — закодированные данные в памяти (WAV, OGG, FLAC, ...)
audio_bytes = open("recording.ogg", "rb").read()
src = BytesSource(audio_bytes, raw=False)
for result in SpeechToText(Engine.VOSK, source=src):
    print(result.text)
```

| Параметр   | По умолчанию | Описание                                                     |
|------------|--------------|--------------------------------------------------------------|
| `data`     | — (обязателен)| `bytes` / `bytearray` / `memoryview` с аудиоданными         |
| `raw`      | `True`       | `True` = raw PCM; `False` = закодированный аудио (WAV, OGG, FLAC, …) |
| `realtime` | `False`      | Подавать чанки в реальном времени или максимально быстро      |
| `block_size`| `4000`      | Сэмплов на чанк                                              |

### Кастомный источник

Наследуйте `AudioSource` для произвольных источников (сеть, виртуальный
кабель и т.д.):

```python
from speech_to_text import AudioSource, SpeechToText

class MySource(AudioSource):
    def start(self) -> None:
        # self.on_audio(chunk)  — отправить PCM-чанк (16 kHz, mono, int16)
        # self.on_finished()    — когда данные закончились (опционально)
        ...

    def stop(self) -> None: ...

    @property
    def is_running(self) -> bool: ...

for result in SpeechToText(Engine.VOSK, source=MySource()):
    print(result.text)
```

## Общие параметры SpeechToText

Эти параметры доступны для любого движка:

```python
from speech_to_text import SpeechToText, Engine

stt = SpeechToText(
    Engine.VOSK,            # Engine enum или строка для кастомных движков
    source=None,            # AudioSource, путь к файлу, или None (микрофон)
    partial_results=True,   # получать промежуточные результаты (по умолчанию True)
    device=None,            # ID или имя микрофона (None = системное по умолчанию)
    sample_rate=16000,      # частота дискретизации (по умолчанию 16000)
    block_size=4000,        # размер аудио-блока (по умолчанию 4000 сэмплов = 250 мс)
)
```

## Модель данных

### TranscriptionResult

Каждый результат распознавания — dataclass:

```python
@dataclass
class TranscriptionResult:
    text: str                        # распознанный текст
    type: ResultType                 # PARTIAL или FINAL
    language: str | None = None      # язык (если определён движком)
    confidence: float | None = None  # уверенность (0.0-1.0)
    timestamp: float = ...           # время получения (time.time())
```

### ResultType

```python
class ResultType(Enum):
    PARTIAL = "partial"  # промежуточный результат — может измениться
    FINAL = "final"      # окончательный результат для фразы
```

**PARTIAL** — текст, распознанный на данный момент. Может быть обновлён
следующим partial- или final-результатом. Полезен для отображения «живого»
текста в UI.

**FINAL** — окончательный результат для законченной фразы. Текст больше
не изменится.

## Кастомные движки

Библиотека поддерживает подключение собственных движков через наследование
и `register_engine`.

### Локальный движок (STTEngine)

```python
from speech_to_text import STTEngine, TranscriptionResult, ResultType, register_engine

class MyLocalEngine(STTEngine):
    def start(self) -> None:
        # инициализация модели
        ...

    def feed_audio(self, chunk: bytes) -> None:
        # chunk — raw PCM, 16 kHz, mono, int16 LE
        text = my_recognize(chunk)
        if text:
            self._on_result(TranscriptionResult(
                text=text,
                type=ResultType.FINAL,
            ))

    def stop(self) -> None:
        # освобождение ресурсов
        ...

register_engine("my_local", MyLocalEngine)
```

### Облачный движок (CloudSTTEngine)

`CloudSTTEngine` наследуется от `STTEngine` и добавляет:

- Валидацию `api_key` (обязательный параметр)
- Готовые атрибуты: `self._api_key`, `self._base_url`, `self._language`,
  `self._sample_rate`, `self._encoding`
- Хелпер `self._emit(text, is_final=True, confidence=None)` для
  отправки результатов одной строкой

```python
from speech_to_text import CloudSTTEngine, register_engine

class MyCloudEngine(CloudSTTEngine):
    def start(self) -> None:
        self._ws = my_connect(self._base_url, key=self._api_key)

    def feed_audio(self, chunk: bytes) -> None:
        resp = self._ws.send_and_recv(chunk)
        self._emit(resp["text"], is_final=resp["done"])

    def stop(self) -> None:
        self._ws.close()

register_engine("my_cloud", MyCloudEngine)
```

После регистрации движок доступен по имени:

```python
for result in SpeechToText("my_cloud", api_key="...", base_url="wss://..."):  # строка для кастомных
    print(result.text)
```

## Архитектура

```
AudioSource                          STTEngine
  ├── MicrophoneSource               ├── VoskEngine      (синхронный)
  ├── FileSource                     ├── WhisperEngine   (отд. поток транскрипции)
  └── <ваш источник>                 ├── DeepgramEngine  (WebSocket)
        │                            └── <ваш движок>
        │                                  │
        └──(audio chunks)──► feed_audio ──►│
                                           ▼
                               TranscriptionResult ──► Queue
                                                         │
                                                   ├──► __iter__()
                                                   └──► on_result()
```

- **AudioSource** — абстрактный источник звука. `MicrophoneSource` захватывает
  с микрофона через `sounddevice` (callback драйвера не блокируется).
  `FileSource` читает аудиофайлы с автоматической конвертацией формата.
- **STTEngine** обрабатывает аудио и вызывает `on_result` callback при
  получении результата.
- **SpeechToText** оркестрирует AudioSource и STTEngine, собирает
  результаты в очередь для итератора и вызывает пользовательские callback'и.
  При окончании файла `FileSource` автоматически останавливает pipeline.

## Структура проекта

```
speech_to_text/
    __init__.py              # публичный API
    core.py                  # SpeechToText — оркестратор
    audio.py                 # AudioSource, MicrophoneSource, FileSource, list_devices
    models.py                # TranscriptionResult, ResultType
    engines/
        __init__.py          # реестр движков, lazy-загрузка
        base.py              # STTEngine, CloudSTTEngine
        vosk_engine.py       # Vosk (Kaldi)
        whisper_engine.py    # faster-whisper + energy-based VAD
        deepgram_engine.py   # Deepgram WebSocket
examples/
    iterator_example.py      # итератор с Vosk
    callback_example.py      # callback с Vosk
    whisper_example.py       # Whisper
    deepgram_example.py      # Deepgram
    custom_engine_example.py     # кастомный движок
    file_source_example.py       # транскрипция из файла
    select_microphone_example.py # выбор микрофона
pyproject.toml                   # зависимости, optional extras
```

## Примеры

Готовые примеры в папке `examples/`:

```bash
# Vosk — итератор
python examples/iterator_example.py

# Vosk — callback
python examples/callback_example.py

# Whisper (нужен pip install -e ".[whisper]")
python examples/whisper_example.py

# Deepgram (нужен DEEPGRAM_API_KEY)
python examples/deepgram_example.py

# Кастомный движок (echo-демо)
python examples/custom_engine_example.py

# Транскрипция аудиофайла
python examples/file_source_example.py recording.wav

# Выбор микрофона из списка
python examples/select_microphone_example.py
```

## Сравнение движков

| Движок   | Задержка     | Точность      | Ресурсы          | Offline | Стоимость            |
|----------|--------------|---------------|------------------|---------|----------------------|
| Vosk     | ~0.3-0.5 с   | средняя       | CPU, ~50 МБ RAM  | да      | бесплатно            |
| Whisper  | ~1-3 с       | высокая       | GPU рекомендуется | да      | бесплатно            |
| Deepgram | ~0.3-1 с     | очень высокая | интернет          | нет     | $200 бесплатно       |

## Лицензия

MIT
