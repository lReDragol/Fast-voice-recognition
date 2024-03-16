import sounddevice as sd  # Библиотека для работы с аудио
import numpy as np  # Библиотека для научных вычислений
import soundfile as sf  # Библиотека для чтения/записи аудиофайлов
import warnings  # Библиотека для управления предупреждениями
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor  # Hugging Face Transformers для моделей глубокого обучения
import torch  # Библиотека для работы с тензорами и глубоким обучением

warnings.filterwarnings("ignore", category=UserWarning)  # Игнорирование предупреждений определённого типа

# Инициализация модели Whisper для распознавания речи
model_id = "openai/whisper-small"  # ID модели Whisper
device = "cuda" if torch.cuda.is_available() else "cpu"  # Выбор устройства для обработки (GPU или CPU)

processor = AutoProcessor.from_pretrained(model_id)  # Инициализация процессора для модели
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,  # Оптимизация использования CPU
    use_safetensors=True,  # Использование SafeTensors для уменьшения потребления памяти
    use_flash_attention_2=False  # Выбор механизма внимания
).to(device)  # Перемещение модели на выбранное устройство

pipe = pipeline(
    "automatic-speech-recognition",  # Тип задачи: автоматическое распознавание речи
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1  # Указание устройства для обработки
)

# Параметры для записи и определения речи
history_length = 5  # Длина истории для анализа громкости
volume_history = np.zeros(history_length)  # История громкости
pre_record_buffer = []  # Буфер для хранения данных до начала записи
record = False  # Флаг записи
all_data = []  # Буфер для хранения записанных данных
record_count = 0  # Счётчик фреймов записи
silence_checks = 0  # Счётчик проверок на тишину
min_silence_checks = 1  # Минимальное количество проверок на тишину для остановки записи
pre_record_length = 2  # Длина буфера предварительной записи в количестве фреймов
file_name = "recorded_speech.wav"  # Имя файла для сохранения записи
samplerate = 44100  # Частота дискретизации аудио
filled_history_count = 0  # Счётчик для определения, когда история громкости заполнена

def process_audio_file(file_path):
    """Обрабатывает аудиофайл и выводит распознанный текст."""
    print("Обработка аудио...")
    result = pipe(file_path)
    print(f"Распознанный текст: {result['text']}")

def sound_detection_callback(indata, frames, time, status):
    """Обратный вызов для обработки аудио потока в реальном времени."""
    global volume_history, record, all_data, record_count, silence_checks, filled_history_count, pre_record_buffer
    volume_norm = np.linalg.norm(indata)  # Вычисление нормы громкости текущего фрейма
    volume_history[:-1] = volume_history[1:]  # Обновление истории громкости
    volume_history[-1] = volume_norm

    if np.all(volume_history > 0):
        filled_history_count += 1

    if filled_history_count < history_length:
        print(f"История громкости заполняется... {1/filled_history_count if filled_history_count else 'Cбор данных'}")
        return

    average_volume = np.mean(volume_history)  # Вычисление средней громкости
    threshold = average_volume * 1.2  # Порог для определения начала говорения

    print(f'Уровень громкости: {volume_norm:.4f}, Среднее: {average_volume:.4f}, Порог: {threshold:.4f}')

    if volume_norm > threshold:
        if not record:
            pre_record_buffer.append(indata.copy())
            if len(pre_record_buffer) > pre_record_length:
                pre_record_buffer.pop(0)
        record_count += 1
        silence_checks = 0
    else:
        if record:
            silence_checks += 1
        else:
            pre_record_buffer = []  # Очистка буфера предварительной записи, если запись не активна
        record_count = max(0, record_count - 1)

    if record_count >= 2 and not record:
        print("Начало записи")
        record = True
        silence_checks = 0
        all_data.extend(pre_record_buffer)  # Добавление данных из буфера предварительной записи
        pre_record_buffer = []

    if record:
        all_data.append(indata.copy())

    if record and silence_checks >= min_silence_checks:
        print("Окончание записи")
        record = False
        if all_data:
            with sf.SoundFile(file_name, mode='w', samplerate=samplerate, channels=indata.shape[1]) as file:
                for data in all_data:
                    file.write(data)
            process_audio_file(file_name)
            all_data = []
        silence_checks = 0

with sd.InputStream(callback=sound_detection_callback, channels=1, samplerate=samplerate, blocksize=int(samplerate * 1)):
    print('Прослушивание микрофона активировано. Говорите...')
    while True:
        sd.sleep(1000)  # Ожидание между итерациями обработки аудио
