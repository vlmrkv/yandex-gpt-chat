import json
import logging
import os
import re
import uuid
import time
from datetime import timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_from_directory

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
app.permanent_session_lifetime = timedelta(days=1)

# Конфигурация Yandex Cloud API
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_BASE_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'

# Проверка наличия необходимых переменных окружения
if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    logger.warning("⚠️ Внимание: YANDEX_API_KEY и YANDEX_FOLDER_ID не установлены в .env файле")

# Модели Yandex Cloud API
MODELS_CONFIG = {
    'yandexgpt-lite': {
        'name': 'YandexGPT Lite',
        'model_uri': 'yandexgpt/latest',
        'cost_per_1k_input_tokens': 0.00015,
        'cost_per_1k_output_tokens': 0.0003,
        'max_tokens': 4000,
        'max_context_tokens': 8000,
        'description': 'Базовая модель Yandex, быстрая и экономичная'
    },
    'yandexgpt': {
        'name': 'YandexGPT Pro',
        'model_uri': 'yandexgpt/latest',
        'cost_per_1k_input_tokens': 0.0006,
        'cost_per_1k_output_tokens': 0.0012,
        'max_tokens': 8000,
        'max_context_tokens': 16000,
        'description': 'Продвинутая модель Yandex с улучшенным качеством'
    },
    # 'qwen': {
    #     'name': 'Qwen 2.5 32B',
    #     'model_uri': 'qwen/qwen-2.5-32b-instruct',
    #     'cost_per_1k_input_tokens': 0.0004,
    #     'cost_per_1k_output_tokens': 0.0008,
    #     'max_tokens': 32000,
    #     'max_context_tokens': 32000,
    #     'description': 'Мощная модель от Alibaba с большим контекстом'
    # },
    # 'gemma': {
    #     'name': 'Gemma 2 9B',
    #     'model_uri': 'gemma/gemma-2-9b-it',
    #     'cost_per_1k_input_tokens': 0.00025,
    #     'cost_per_1k_output_tokens': 0.0005,
    #     'max_tokens': 16000,
    #     'max_context_tokens': 16000,
    #     'description': 'Эффективная модель от Google, оптимизированная для диалогов'
    # }
}

# Конфигурация компрессии
COMPRESSION_CONFIG = {
    'enable_compression': True,  # Включить компрессию
    'compression_interval': 10,  # Сжимать каждые 10 сообщений
    'summary_model': 'yandexgpt-lite',  # Модель для создания summary
    'max_summary_tokens': 300,  # Максимальная длина summary
    'keep_last_messages': 3,  # Оставлять последние N сообщений несжатыми
    'min_compression_saving': 0.3,  # Минимальная экономия токенов для компрессии (30%)
}

# Базовые системные промпты для разных режимов
SYSTEM_PROMPTS = {
    'default': 'Ты полезный ассистент. Отвечай вежливо и по существу.',
    'json_format': """Ты полезный ассистент. Всегда отвечай в формате JSON.""",
    'tz_collection': """Ты - профессиональный аналитик, который собирает требования для Технического Задания.""",

    # Промпт для компрессии истории
    'history_compression': """Ты - эксперт по сжатию информации. Твоя задача - создать краткое изложение диалога, сохранив ключевые моменты, решения, важные детали и контекст.

ПРАВИЛА СОЗДАНИЯ SUMMARY:
1. Сохрани основную тему/цель диалога
2. Сохрани ключевые решения и договоренности
3. Сохрани важные факты, цифры, даты
4. Сохрани контекст (кто, что, зачем, когда)
5. Игнорируй приветствия, прощания, технические детали
6. Будь максимально кратким, но информативным
7. Пиши в формате: "Обсуждение [тема]. Решено: ... Важно: ..."

ФОРМАТ ОТВЕТА:
Только summary без дополнительных комментариев."""
}


def init_session():
    """Инициализация сессии с настройками по умолчанию"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'compressed_history' not in session:
        session['compressed_history'] = []
    if 'chat_mode' not in session:
        session['chat_mode'] = 'default'
    if 'tz_data' not in session:
        session['tz_data'] = None
    if 'tz_complete' not in session:
        session['tz_complete'] = False
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    if 'custom_system_prompt' not in session:
        session['custom_system_prompt'] = None
    if 'current_role_name' not in session:
        session['current_role_name'] = 'Обычный ассистент'
    if 'last_system_prompt' not in session:
        session['last_system_prompt'] = SYSTEM_PROMPTS['default']
    if 'temperature' not in session:
        session['temperature'] = "0.7"
    if 'selected_model' not in session:
        session['selected_model'] = 'yandexgpt-lite'
    # Статистика токенов
    if 'total_input_tokens' not in session:
        session['total_input_tokens'] = 0
    if 'total_output_tokens' not in session:
        session['total_output_tokens'] = 0
    if 'total_cost' not in session:
        session['total_cost'] = 0.0
    # Статистика компрессии
    if 'compression_stats' not in session:
        session['compression_stats'] = {
            'total_compressions': 0,
            'tokens_saved': 0,
            'original_tokens': 0,
            'compressed_tokens': 0,
            'compression_ratio': 0.0
        }
    if 'last_compression_message_id' not in session:
        session['last_compression_message_id'] = -1


def get_model_config(model_key):
    """Получить конфигурацию модели по ключу"""
    return MODELS_CONFIG.get(model_key, MODELS_CONFIG['yandexgpt-lite'])


def estimate_tokens(text):
    """Примерная оценка количества токенов в тексте"""
    if not text:
        return 0
    # Более точная оценка: 1 токен ≈ 4 символа
    return max(1, int(len(text) / 4))


def analyze_response_behavior(input_tokens, output_tokens, temperature, response_text, model_config):
    """Анализ поведения модели на основе токенов и температуры"""
    # Убедимся, что все значения являются числами
    try:
        input_tokens = int(input_tokens) if input_tokens is not None else 0
        output_tokens = int(output_tokens) if output_tokens is not None else 0
        temperature = float(temperature) if temperature is not None else 0.7
    except (ValueError, TypeError) as e:
        logger.error(f"Ошибка преобразования типов в analyze_response_behavior: {e}")
        input_tokens = 0
        output_tokens = 0
        temperature = 0.7

    behavior_analysis = {
        'efficiency_score': 0,
        'verbosity_level': 'нормальный',
        'temperature_effect': 'стандартный',
        'structure': 'стандартная',
        'recommendations': [],
        'token_ratio': 0
    }

    # Рассчитываем соотношение токенов
    if input_tokens > 0:
        token_ratio = output_tokens / input_tokens
        behavior_analysis['token_ratio'] = round(token_ratio, 2)

        # Оценка эффективности
        if input_tokens < 50:  # Короткий запрос
            if output_tokens < 30:
                behavior_analysis['efficiency_score'] = 30
            elif output_tokens > 500:
                behavior_analysis['efficiency_score'] = 90
            else:
                behavior_analysis['efficiency_score'] = 70
        else:  # Длинный запрос
            if token_ratio > 2:
                behavior_analysis['efficiency_score'] = 90
            elif token_ratio > 1:
                behavior_analysis['efficiency_score'] = 80
            elif token_ratio > 0.5:
                behavior_analysis['efficiency_score'] = 60
            else:
                behavior_analysis['efficiency_score'] = 40

    # Уровень многословности
    if output_tokens < 30:
        behavior_analysis['verbosity_level'] = 'очень краткий'
    elif output_tokens < 100:
        behavior_analysis['verbosity_level'] = 'краткий'
    elif output_tokens < 300:
        behavior_analysis['verbosity_level'] = 'нормальный'
    elif output_tokens < 800:
        behavior_analysis['verbosity_level'] = 'подробный'
    else:
        behavior_analysis['verbosity_level'] = 'очень подробный'

    # Влияние температуры
    if temperature < 0.2:
        behavior_analysis['temperature_effect'] = 'очень детерминированный'
    elif temperature < 0.4:
        behavior_analysis['temperature_effect'] = 'детерминированный'
    elif temperature < 0.7:
        behavior_analysis['temperature_effect'] = 'сбалансированный'
    elif temperature < 0.9:
        behavior_analysis['temperature_effect'] = 'креативный'
    else:
        behavior_analysis['temperature_effect'] = 'очень креативный'

    # Анализ структуры ответа
    if not response_text:
        response_text = ""

    lines_count = response_text.count('\n') + 1
    sentences_count = response_text.count('.') + response_text.count('!') + response_text.count('?')

    if lines_count > 10 and sentences_count > 5:
        behavior_analysis['structure'] = 'хорошо структурированный'
    elif '```' in response_text or '```json' in response_text:
        behavior_analysis['structure'] = 'код/структурированный'
    elif response_text.count('\n') > 5:
        behavior_analysis['structure'] = 'многострочный'
    elif len(response_text.split()) < 50:
        behavior_analysis['structure'] = 'конспективный'
    else:
        behavior_analysis['structure'] = 'сплошной текст'

    # Рекомендации
    if behavior_analysis.get('token_ratio', 0) > 3:
        behavior_analysis['recommendations'].append('Модель генерирует очень подробные ответы')
    elif behavior_analysis.get('token_ratio', 0) < 0.3:
        behavior_analysis['recommendations'].append('Ответы слишком кратки, попробуйте увеличить max_tokens')

    if temperature < 0.3 and output_tokens < 100:
        behavior_analysis['recommendations'].append('Низкая температура делает ответы краткими')
    elif temperature > 0.8 and output_tokens > 500:
        behavior_analysis['recommendations'].append('Высокая температура может создавать избыточный текст')

    return behavior_analysis


def call_yandex_api(model_config, messages, temperature, max_tokens):
    """Вызов Yandex Cloud API с замером времени и метрик"""
    headers = {
        'Authorization': f'Api-Key {YANDEX_API_KEY}',
        'Content-Type': 'application/json'
    }

    payload = {
        'modelUri': f'gpt://{YANDEX_FOLDER_ID}/{model_config["model_uri"]}',
        'completionOptions': {
            'stream': False,
            'temperature': float(temperature),
            'maxTokens': int(max_tokens)
        },
        'messages': messages
    }

    # Оцениваем количество входных токенов
    input_text = ' '.join([msg['text'] for msg in messages])
    estimated_input_tokens = estimate_tokens(input_text)

    # Замер времени выполнения
    start_time = time.time()
    response = requests.post(YANDEX_API_BASE_URL, headers=headers, json=payload, timeout=60)
    end_time = time.time()
    execution_time = end_time - start_time

    if response.status_code != 200:
        logger.error(f"API Error {response.status_code}: {response.text}")
        return None, execution_time, estimated_input_tokens, 0, 0.0

    result = response.json()

    # Получаем количество токенов из ответа
    input_tokens = estimated_input_tokens
    output_tokens = 0

    try:
        # Извлекаем ответ
        response_text = ""
        if 'result' in result and 'alternatives' in result['result']:
            response_text = result['result']['alternatives'][0]['message']['text']
        elif 'alternatives' in result:
            response_text = result['alternatives'][0]['message']['text']
        else:
            response_text = extract_message_from_response(result)

        # Оцениваем выходные токены
        output_tokens = estimate_tokens(response_text)

        # Пытаемся получить точные значения из API
        if 'result' in result and 'usage' in result['result']:
            usage = result['result']['usage']
            input_tokens = int(usage.get('inputTextTokens', estimated_input_tokens))
            output_tokens = int(usage.get('completionTokens', output_tokens))
        elif 'usage' in result:
            usage = result['usage']
            input_tokens = int(usage.get('inputTextTokens', estimated_input_tokens))
            output_tokens = int(usage.get('completionTokens', output_tokens))

    except (KeyError, IndexError, ValueError, TypeError) as e:
        logger.error(f"Ошибка при обработке ответа API: {e}")
        # Используем оценки
        output_tokens = estimate_tokens(extract_message_from_response(result))

    # Рассчитываем стоимость
    try:
        input_cost = (input_tokens / 1000.0) * float(model_config['cost_per_1k_input_tokens'])
        output_cost = (output_tokens / 1000.0) * float(model_config['cost_per_1k_output_tokens'])
        total_cost = input_cost + output_cost
    except (TypeError, ValueError) as e:
        logger.error(f"Ошибка при расчете стоимости: {e}")
        total_cost = 0.0

    return result, execution_time, input_tokens, output_tokens, total_cost


def compress_history(messages_to_compress, model_key='yandexgpt-lite'):
    """Создание summary для сжатия истории диалога"""
    if not messages_to_compress or len(messages_to_compress) < 2:
        return "", 0, 0, 0.0

    # Подготавливаем сообщения для компрессии
    compression_messages = [
        {'role': 'system', 'text': SYSTEM_PROMPTS['history_compression']},
        {'role': 'user',
         'text': f"Создай краткое изложение этого диалога:\n\n{format_messages_for_compression(messages_to_compress)}"}
    ]

    # Получаем конфигурацию модели
    model_config = get_model_config(model_key)

    # Вызываем API для создания summary
    result, execution_time, input_tokens, output_tokens, cost = call_yandex_api(
        model_config=model_config,
        messages=compression_messages,
        temperature=0.3,  # Низкая температура для более точного summary
        max_tokens=COMPRESSION_CONFIG['max_summary_tokens']
    )

    if result is None:
        return "", 0, 0, 0.0

    # Извлекаем summary
    summary = ""
    try:
        if 'result' in result and 'alternatives' in result['result']:
            summary = result['result']['alternatives'][0]['message']['text']
        elif 'alternatives' in result:
            summary = result['alternatives'][0]['message']['text']
    except (KeyError, IndexError) as e:
        logger.error(f"Ошибка при извлечении summary: {e}")
        summary = "Не удалось создать summary"

    return summary.strip(), input_tokens, output_tokens, cost


def format_messages_for_compression(messages):
    """Форматирование сообщений для компрессии"""
    formatted = []
    for msg in messages:
        role = "Пользователь" if msg['role'] == 'user' else "Ассистент"
        formatted.append(f"{role}: {msg['text']}")
    return "\n".join(formatted)


def should_compress_history(full_history):
    """Определить, нужно ли сжимать историю"""
    if not COMPRESSION_CONFIG['enable_compression']:
        return False

    # Проверяем по количеству сообщений
    if len(full_history) >= COMPRESSION_CONFIG['compression_interval']:
        return True

    # Проверяем по количеству токенов (опционально)
    total_tokens = sum(estimate_tokens(msg['text']) for msg in full_history)
    if total_tokens > 2000:  # Если больше 2000 токенов
        return True

    return False


def get_compression_candidate(full_history, last_compressed_id):
    """Получить сообщения для компрессии"""
    keep_last = COMPRESSION_CONFIG['keep_last_messages']

    # Ищем сообщения, которые еще не были сжаты
    uncompressed_messages = []
    for i, msg in enumerate(full_history):
        if i <= last_compressed_id:
            continue
        # Исключаем последние сообщения
        if i >= len(full_history) - keep_last:
            break
        uncompressed_messages.append(msg)

    return uncompressed_messages


def update_compression_stats(original_tokens, compressed_tokens):
    """Обновить статистику компрессии"""
    if 'compression_stats' not in session:
        session['compression_stats'] = {
            'total_compressions': 0,
            'tokens_saved': 0,
            'original_tokens': 0,
            'compressed_tokens': 0,
            'compression_ratio': 0.0
        }

    stats = session['compression_stats']
    stats['total_compressions'] += 1
    stats['original_tokens'] += original_tokens
    stats['compressed_tokens'] += compressed_tokens

    if original_tokens > 0:
        tokens_saved = original_tokens - compressed_tokens
        stats['tokens_saved'] += tokens_saved
        compression_ratio = compressed_tokens / original_tokens
        stats['compression_ratio'] = round(compression_ratio * 100, 2)

    session['compression_stats'] = stats


def get_effective_history():
    """Получить эффективную историю (сжатая + последние сообщения)"""
    full_history = session.get('chat_history', [])
    compressed_history = session.get('compressed_history', [])
    keep_last = COMPRESSION_CONFIG['keep_last_messages']

    # Если нет сжатой истории, возвращаем полную историю
    if not compressed_history:
        return full_history[-keep_last * 2:] if len(full_history) > keep_last * 2 else full_history

    # Объединяем сжатую историю с последними сообщениями
    last_messages = full_history[-keep_last:] if len(full_history) > keep_last else full_history

    effective_history = []

    # Добавляем сжатую историю
    for summary in compressed_history:
        effective_history.append({
            'role': 'system',
            'text': f"📚 Краткое содержание предыдущего диалога: {summary}"
        })

    # Добавляем последние сообщения
    effective_history.extend(last_messages)

    return effective_history


@app.route('/')
def index():
    """Главная страница с чатом"""
    init_session()
    return render_template('index.html',
                           history=session['chat_history'],
                           chat_mode=session['chat_mode'],
                           tz_complete=session['tz_complete'],
                           temperature=session['temperature'],
                           current_role=session.get('current_role_name', 'Обычный ассистент'),
                           models=MODELS_CONFIG,
                           selected_model=session.get('selected_model', 'yandexgpt-lite'),
                           total_input_tokens=session.get('total_input_tokens', 0),
                           total_output_tokens=session.get('total_output_tokens', 0),
                           total_cost=session.get('total_cost', 0),
                           compression_stats=session.get('compression_stats', {}))


@app.route('/send_message', methods=['POST'])
def send_message():
    """Обработка отправки сообщения"""
    try:
        logger.info(f"Получен запрос: {request.json}")
        user_message = request.json.get('message', '').strip()
        require_json = request.json.get('require_json', False)
        model_key = request.json.get('model', 'yandexgpt-lite')

        # Получаем температуру из запроса
        temperature = request.json.get('temperature')
        if temperature is not None:
            try:
                temperature_val = float(temperature)
                temperature_val = max(0.0, min(1.0, temperature_val))
            except (ValueError, TypeError):
                temperature_val = float(session.get('temperature', 0.7))
        else:
            temperature_val = float(session.get('temperature', 0.7))

        if not user_message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400

        init_session()

        # Сохраняем температуру и модель в сессии
        session['temperature'] = str(temperature_val)
        session['selected_model'] = model_key

        # Проверяем, является ли сообщение командой
        if user_message.lower().startswith('/role '):
            return handle_role_command(user_message)
        elif user_message.lower().startswith('/system '):
            return handle_system_command(user_message)
        elif user_message.lower() in ['/role', '/current', '/whoami']:
            return show_current_role()
        elif user_message.lower() in ['/reset', '/default', '/clearrole']:
            return reset_role()
        elif user_message.lower() in ['/help', '/commands', '/?']:
            return show_help()
        elif user_message.lower() == '/stats':
            return show_stats()
        elif user_message.lower() == '/compression_stats':
            return show_compression_stats()

        # Получаем текущий режим чата
        chat_mode = session.get('chat_mode', 'default')
        tz_complete = session.get('tz_complete', False)

        # Если сбор ТЗ завершен, но не в режиме ТЗ - сбрасываем флаг
        if tz_complete and chat_mode != 'tz_collection':
            session['tz_complete'] = False
            tz_complete = False

        # Если сбор ТЗ завершен, не принимаем новые сообщения
        if tz_complete:
            return jsonify({
                'error': 'Сбор ТЗ завершен. Начните новый сбор или очистите историю.',
                'tz_data': session.get('tz_data')
            }), 400

        # Определяем фактический режим работы
        if require_json and chat_mode not in ['tz_collection']:
            actual_mode = 'json_format'
        else:
            actual_mode = chat_mode

        # Получаем полную историю диалога из сессии
        full_history = session.get('chat_history', [])

        # Добавляем сообщение пользователя в полную историю
        full_history.append({'role': 'user', 'text': user_message})

        # Проверяем, нужно ли сжимать историю
        compression_occurred = False
        compression_cost = 0.0
        compression_input_tokens = 0
        compression_output_tokens = 0

        if COMPRESSION_CONFIG['enable_compression'] and should_compress_history(full_history):
            # Получаем сообщения для компрессии
            candidate_messages = get_compression_candidate(
                full_history,
                session.get('last_compression_message_id', -1)
            )

            if candidate_messages and len(candidate_messages) >= 2:
                # Создаем summary
                summary, comp_input, comp_output, comp_cost = compress_history(
                    candidate_messages,
                    COMPRESSION_CONFIG['summary_model']
                )

                if summary:
                    # Добавляем summary в сжатую историю
                    compressed_history = session.get('compressed_history', [])
                    compressed_history.append(summary)
                    session['compressed_history'] = compressed_history

                    # Обновляем ID последнего сжатого сообщения
                    last_msg_index = full_history.index(candidate_messages[-1])
                    session['last_compression_message_id'] = last_msg_index

                    # Обновляем статистику компрессии
                    original_tokens = sum(estimate_tokens(msg['text']) for msg in candidate_messages)
                    compressed_tokens = estimate_tokens(summary)
                    update_compression_stats(original_tokens, compressed_tokens)

                    compression_occurred = True
                    compression_cost = comp_cost
                    compression_input_tokens = comp_input
                    compression_output_tokens = comp_output

                    logger.info(
                        f"✅ Сжатие истории: {len(candidate_messages)} сообщений -> summary ({compressed_tokens} токенов)")
                    logger.info(f"   Экономия: {original_tokens - compressed_tokens} токенов")

        # Получаем эффективную историю (сжатая + последние сообщения)
        effective_history = get_effective_history()

        # Выбираем системный промпт в зависимости от режима
        if session.get('custom_system_prompt'):
            system_prompt = session['custom_system_prompt']
        else:
            system_prompt = SYSTEM_PROMPTS.get(actual_mode, SYSTEM_PROMPTS['default'])

        # Сохраняем текущий промпт для будущих запросов
        session['last_system_prompt'] = system_prompt

        # Подготавливаем сообщения для API
        messages = [{'role': 'system', 'text': system_prompt}]

        # Добавляем эффективную историю диалога
        max_messages = 20 if actual_mode == 'tz_collection' else 15
        filtered_history = []
        for msg in effective_history:
            if msg['role'] in ['user', 'assistant']:
                filtered_history.append(msg)

        for msg in filtered_history[-max_messages:]:
            messages.append({
                'role': msg['role'],
                'text': msg['text']
            })

        # Получаем конфигурацию выбранной модели
        model_config = get_model_config(model_key)

        # Настраиваем параметры в зависимости от режима
        if actual_mode == 'json_format':
            max_tokens = min(1500, int(model_config['max_tokens']))
        elif actual_mode == 'tz_collection':
            max_tokens = min(2000, int(model_config['max_tokens']))
        elif session.get('custom_system_prompt'):
            max_tokens = min(2000, int(model_config['max_tokens']))
        else:
            max_tokens = min(1500, int(model_config['max_tokens']))

        # Вызываем API с замером метрик
        result, execution_time, input_tokens, output_tokens, cost = call_yandex_api(
            model_config, messages, temperature_val, max_tokens
        )

        if result is None:
            return jsonify({'error': 'Ошибка при вызове API модели'}), 500

        # Получаем ответ ассистента
        assistant_message = ""
        try:
            if 'result' in result and 'alternatives' in result['result']:
                assistant_message = result['result']['alternatives'][0]['message']['text']
            elif 'alternatives' in result:
                assistant_message = result['alternatives'][0]['message']['text']
            else:
                assistant_message = extract_message_from_response(result)
        except (KeyError, IndexError) as e:
            logger.error(f"Ошибка при извлечении ответа: {e}")
            assistant_message = "Не удалось получить ответ от модели."

        # Убедимся, что токены - целые числа
        try:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
        except (ValueError, TypeError):
            input_tokens = estimate_tokens(' '.join([msg['text'] for msg in messages]))
            output_tokens = estimate_tokens(assistant_message)

        # Добавляем токены компрессии к общим токенам
        total_input = input_tokens + compression_input_tokens
        total_output = output_tokens + compression_output_tokens
        total_cost_api = cost + compression_cost

        # Анализируем поведение модели на основе токенов и температуры
        behavior_analysis = analyze_response_behavior(
            input_tokens, output_tokens, temperature_val, assistant_message, model_config
        )

        # Обновляем общую статистику токенов
        session['total_input_tokens'] = session.get('total_input_tokens', 0) + total_input
        session['total_output_tokens'] = session.get('total_output_tokens', 0) + total_output
        session['total_cost'] = session.get('total_cost', 0.0) + total_cost_api

        # Добавляем ответ ассистента в полную историю
        full_history.append({'role': 'assistant', 'text': assistant_message})

        # Ограничиваем полную историю (максимум 100 сообщений)
        if len(full_history) > 100:
            full_history = full_history[-100:]

        # Сохраняем обновленную историю в сессии
        session['chat_history'] = full_history
        session.modified = True

        # Формируем ответ для клиента
        response_data = {
            'response': assistant_message,
            'history': full_history,
            'chat_mode': chat_mode,
            'temperature': temperature_val,
            'require_json': require_json,
            'tz_complete': session.get('tz_complete', False),
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(full_history),
            'model': model_key,
            'model_name': model_config['name'],
            'execution_time': round(execution_time, 2),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost': round(cost, 5),
            'behavior_analysis': behavior_analysis,
            'token_stats': {
                'total_input': session['total_input_tokens'],
                'total_output': session['total_output_tokens'],
                'total_cost': session['total_cost']
            },
            'compression': {
                'occurred': compression_occurred,
                'cost': round(compression_cost, 5),
                'input_tokens': compression_input_tokens,
                'output_tokens': compression_output_tokens
            }
        }

        if compression_occurred:
            response_data['compression_stats'] = session.get('compression_stats', {})

        logger.info(f"Модель: {model_key}, Вход: {input_tokens}, Выход: {output_tokens}, "
                    f"Время: {execution_time:.2f}с, Стоимость: {cost:.5f} руб")
        if compression_occurred:
            logger.info(f"Сжатие: +{compression_input_tokens} входных, +{compression_output_tokens} выходных токенов")

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({'error': 'Ошибка соединения с YandexGPT API.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500


def extract_message_from_response(result):
    """Извлечь сообщение из ответа API при нестандартной структуре"""
    try:
        if 'result' in result and 'alternatives' in result['result']:
            return result['result']['alternatives'][0]['message']['text']

        result_str = json.dumps(result)
        if '"text":' in result_str:
            match = re.search(r'"text":\s*"([^"]+)"', result_str)
            if match:
                return match.group(1)

        return "Ответ получен, но не удалось распарсить структуру."
    except:
        return "Ошибка при обработке ответа от модели."


def handle_role_command(user_message):
    """Обработка команды для смены роли"""
    try:
        role_text = user_message[6:].strip()

        predefined_roles = {
            'инженер': {
                'prompt': 'Ты — инженер-прагматик с 20-летним стажем.',
                'name': 'Инженер'
            },
            'режиссер': {
                'prompt': 'Ты — знаменитый режиссёр с безграничной фантазией.',
                'name': 'Режиссер'
            },
            'бабушка': {
                'prompt': 'Ты — добрая, мудрая бабушка, которая повидала многое на своём веку.',
                'name': 'Бабушка'
            }
        }

        matched_role = None
        role_name = None

        for key, value in predefined_roles.items():
            if key.startswith(role_text.lower()) or role_text.lower() in key:
                matched_role = value
                role_name = value['name']
                break

        if not matched_role:
            available_roles = ", ".join(predefined_roles.keys())
            return jsonify({
                'response': f"❌ Роль '{role_text}' не найдена. Доступные роли: {available_roles}",
                'is_command': True,
                'current_role': session.get('current_role_name', 'Обычный ассистент'),
                'history_count': len(session.get('chat_history', []))
            })

        session['custom_system_prompt'] = matched_role['prompt']
        session['chat_mode'] = 'custom'
        session['current_role_name'] = role_name

        chat_history = session.get('chat_history', [])
        system_message = f"🔄 Роль изменена на: **{role_name}**"
        chat_history.append({'role': 'system', 'text': system_message})
        session['chat_history'] = chat_history
        session.modified = True

        return jsonify({
            'response': system_message,
            'history': chat_history,
            'is_command': True,
            'current_role': role_name,
            'history_count': len(chat_history)
        })

    except Exception as e:
        logger.error(f"Error handling role command: {str(e)}")
        return jsonify({
            'response': f"❌ Ошибка при смене роли: {str(e)}",
            'is_command': True,
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(session.get('chat_history', []))
        })


def handle_system_command(user_message):
    """Обработка команды для установки произвольного системного промпта"""
    try:
        system_prompt = user_message[8:].strip()

        if not system_prompt:
            return jsonify({
                'response': "❌ Укажите системный промпт",
                'is_command': True,
                'current_role': session.get('current_role_name', 'Обычный ассистент'),
                'history_count': len(session.get('chat_history', []))
            })

        session['custom_system_prompt'] = system_prompt
        session['chat_mode'] = 'custom'
        session['current_role_name'] = "Кастомная роль"

        chat_history = session.get('chat_history', [])
        system_message = f"🔄 Установлен кастомный системный промпт"
        chat_history.append({'role': 'system', 'text': system_message})
        session['chat_history'] = chat_history
        session.modified = True

        return jsonify({
            'response': system_message,
            'history': chat_history,
            'is_command': True,
            'current_role': "Кастомная роль",
            'history_count': len(chat_history)
        })

    except Exception as e:
        logger.error(f"Error handling system command: {str(e)}")
        return jsonify({
            'response': f"❌ Ошибка при установке системного промпта: {str(e)}",
            'is_command': True,
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(session.get('chat_history', []))
        })


def show_current_role():
    """Показать текущую роль"""
    current_role = session.get('current_role_name', 'Обычный ассистент')
    return jsonify({
        'response': f"👤 **Текущая роль:** {current_role}",
        'is_command': True,
        'current_role': current_role,
        'history_count': len(session.get('chat_history', []))
    })


def reset_role():
    """Сбросить роль к значениям по умолчанию"""
    session['custom_system_prompt'] = None
    session['chat_mode'] = 'default'
    session['current_role_name'] = 'Обычный ассистент'

    chat_history = session.get('chat_history', [])
    system_message = "🔄 Роль сброшена к значениям по умолчанию"
    chat_history.append({'role': 'system', 'text': system_message})
    session['chat_history'] = chat_history
    session.modified = True

    return jsonify({
        'response': system_message,
        'history': chat_history,
        'is_command': True,
        'current_role': 'Обычный ассистент',
        'history_count': len(chat_history)
    })


def show_stats():
    """Показать статистику использования токенов"""
    total_input = session.get('total_input_tokens', 0)
    total_output = session.get('total_output_tokens', 0)
    total_cost = session.get('total_cost', 0.0)
    compression_stats = session.get('compression_stats', {})

    stats_text = f"""
📊 **Статистика использования токенов:**

• Всего входных токенов: {total_input}
• Всего выходных токенов: {total_output}
• Всего токенов: {total_input + total_output}
• Общая стоимость: {total_cost:.5f} руб

📦 **Статистика компрессии:**
• Сжатий выполнено: {compression_stats.get('total_compressions', 0)}
• Токенов сэкономлено: {compression_stats.get('tokens_saved', 0)}
• Коэффициент сжатия: {compression_stats.get('compression_ratio', 0)}%
"""

    return jsonify({
        'response': stats_text,
        'is_command': True,
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'history_count': len(session.get('chat_history', []))
    })


def show_compression_stats():
    """Показать детальную статистику компрессии"""
    compression_stats = session.get('compression_stats', {})
    config = COMPRESSION_CONFIG

    stats_text = f"""
🔧 **Конфигурация компрессии:**
• Включена: {'✅ Да' if config['enable_compression'] else '❌ Нет'}
• Интервал сжатия: каждые {config['compression_interval']} сообщений
• Модель для summary: {config['summary_model']}
• Оставлять сообщений: {config['keep_last_messages']}
• Макс. токенов summary: {config['max_summary_tokens']}

📈 **Статистика компрессии:**
• Всего сжатий: {compression_stats.get('total_compressions', 0)}
• Исходных токенов: {compression_stats.get('original_tokens', 0)}
• Сжатых токенов: {compression_stats.get('compressed_tokens', 0)}
• Токенов сэкономлено: {compression_stats.get('tokens_saved', 0)}
• Коэффициент сжатия: {compression_stats.get('compression_ratio', 0)}%

💡 **Эффективность:**
{'- Компрессия работает эффективно' if compression_stats.get('compression_ratio', 100) < 50 else '- Компрессия может быть улучшена'}
"""

    return jsonify({
        'response': stats_text,
        'is_command': True,
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'history_count': len(session.get('chat_history', []))
    })


def show_help():
    """Показать справку по командам"""
    help_text = """
📚 **Доступные команды:**

/role [инженер|режиссер|бабушка] - сменить роль
/system [промпт] - установить произвольный промпт
/reset - сбросить роль
/stats - показать статистику токенов
/compression_stats - показать статистику компрессии
/help - показать справку
"""
    return jsonify({
        'response': help_text,
        'is_command': True,
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'history_count': len(session.get('chat_history', []))
    })


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Установка режима работы чата"""
    try:
        mode = request.json.get('mode', 'default')
        valid_modes = ['default', 'json_format', 'tz_collection']

        if mode not in valid_modes:
            return jsonify({'error': 'Неверный режим'}), 400

        init_session()

        if mode != 'tz_collection':
            session['tz_complete'] = False
            session['tz_data'] = None

        if mode == 'tz_collection':
            session['chat_history'] = [{
                'role': 'assistant',
                'text': 'Здравствуйте! Я помогу вам составить Техническое Задание. Расскажите о вашем проекте.'
            }]
            # Сбрасываем сжатую историю при смене режима
            session['compressed_history'] = []
            session['last_compression_message_id'] = -1
        else:
            session['custom_system_prompt'] = None
            session['current_role_name'] = 'Обычный ассистент'

        session['chat_mode'] = mode
        session.modified = True

        return jsonify({
            'success': True,
            'mode': mode,
            'message': f'Режим изменен на: {mode}',
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history': session.get('chat_history', [])
        })
    except Exception as e:
        logger.error(f"Error setting mode: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Очистка истории диалога"""
    init_session()
    session['chat_history'] = []
    session['compressed_history'] = []
    session['tz_data'] = None
    session['tz_complete'] = False
    # Сбрасываем статистику
    session['total_input_tokens'] = 0
    session['total_output_tokens'] = 0
    session['total_cost'] = 0.0
    # Сбрасываем статистику компрессии
    session['compression_stats'] = {
        'total_compressions': 0,
        'tokens_saved': 0,
        'original_tokens': 0,
        'compressed_tokens': 0,
        'compression_ratio': 0.0
    }
    session['last_compression_message_id'] = -1
    session.modified = True

    system_message = "🗑️ История диалога, компрессия и статистика очищены."

    return jsonify({
        'success': True,
        'response': system_message,
        'history': [{'role': 'system', 'text': system_message}],
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'history_count': 0,
        'token_stats': {
            'total_input': 0,
            'total_output': 0,
            'total_cost': 0
        }
    })


@app.route('/toggle_compression', methods=['POST'])
def toggle_compression():
    """Включение/выключение компрессии"""
    init_session()

    # Получаем состояние из запроса или переключаем
    enable = request.json.get('enable')
    if enable is None:
        # Переключаем текущее состояние
        COMPRESSION_CONFIG['enable_compression'] = not COMPRESSION_CONFIG['enable_compression']
    else:
        COMPRESSION_CONFIG['enable_compression'] = bool(enable)

    status = "включена" if COMPRESSION_CONFIG['enable_compression'] else "выключена"

    return jsonify({
        'success': True,
        'enable_compression': COMPRESSION_CONFIG['enable_compression'],
        'message': f'Компрессия истории {status}'
    })


@app.route('/api_info')
def api_info():
    """Информация о статусе API"""
    status = 'configured' if YANDEX_API_KEY and YANDEX_FOLDER_ID else 'not_configured'
    return jsonify({
        'status': status,
        'has_api_key': bool(YANDEX_API_KEY),
        'has_folder_id': bool(YANDEX_FOLDER_ID),
        'compression_enabled': COMPRESSION_CONFIG['enable_compression']
    })


@app.route('/get_session_info', methods=['GET'])
def get_session_info():
    """Получить информацию о текущей сессии"""
    init_session()
    chat_history = session.get('chat_history', [])

    return jsonify({
        'chat_mode': session.get('chat_mode', 'default'),
        'temperature': session.get('temperature', 0.7),
        'tz_complete': session.get('tz_complete', False),
        'tz_data': session.get('tz_data'),
        'history_length': len(chat_history),
        'compressed_history_length': len(session.get('compressed_history', [])),
        'session_id': session.get('session_id'),
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'has_custom_prompt': bool(session.get('custom_system_prompt')),
        'selected_model': session.get('selected_model', 'yandexgpt-lite'),
        'token_stats': {
            'total_input': session.get('total_input_tokens', 0),
            'total_output': session.get('total_output_tokens', 0),
            'total_cost': session.get('total_cost', 0.0)
        },
        'compression_stats': session.get('compression_stats', {}),
        'compression_enabled': COMPRESSION_CONFIG['enable_compression']
    })


@app.route('/get_models', methods=['GET'])
def get_models():
    """Получить список доступных моделей"""
    return jsonify(MODELS_CONFIG)


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Сервис для обслуживания статических файлов"""
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    # Создаем папки если их нет
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)