import json
import logging
import os
import re
import uuid
from datetime import timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, app

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
app.permanent_session_lifetime = timedelta(days=1)

# Конфигурация YandexGPT API
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')
YANDEX_API_URL = 'https://llm.api.cloud.yandex.net/foundationModels/v1/completion'

# Проверка наличия необходимых переменных окружения
if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    logger.warning("⚠️ Внимание: YANDEX_API_KEY и YANDEX_FOLDER_ID не установлены в .env файле")

# Системные промпты для разных режимов
SYSTEM_PROMPTS = {
    'default': 'Ты полезный ассистент. Отвечай вежливо и по существу.',

    'json_format': """Ты полезный ассистент. Всегда отвечай в формате JSON со следующей структурой:
    {
        "response": "Твой основной ответ пользователю",
        "sentiment": "нейтральный/положительный/отрицательный",
        "confidence": 0.95,
        "entities": ["список", "извлеченных", "сущностей"],
        "intent": "намерение_пользователя",
        "suggestions": ["предложение1", "предложение2"]
    }

    Правила:
    1. Всегда возвращай валидный JSON
    2. Поле "response" должно содержать текстовый ответ пользователю
    3. Поле "sentiment" должно быть одним из: "положительный", "нейтральный", "отрицательный"
    4. Поле "confidence" - число от 0 до 1 (уверенность в ответе)
    5. Поле "entities" - массив строк с ключевыми сущностями из запроса
    6. Поле "intent" - распознанное намерение пользователя
    7. Поле "suggestions" - массив предложений для продолжения диалога

    Не добавляй никаких дополнительных комментариев кроме JSON!""",

    'tz_collection': """Ты - профессиональный аналитик, который собирает требования для Технического Задания (ТЗ).

ТВОЯ ЗАДАЧА: В ходе диалога с заказчиком собрать всю необходимую информацию и сформировать финальное ТЗ в формате JSON.

ПРАВИЛА РАБОТЫ:
1. Задавай уточняющие вопросы по очереди, чтобы понять:
   - Цели и задачи проекта
   - Целевую аудиторию
   - Функциональные требования
   - Нефункциональные требования
   - Ограничения и допущения
   - Сроки и бюджет

2. КОГДА ИНФОРМАЦИЯ ПОЛНОСТЬЮ СОБРАНА (ты задал все важные вопросы и получил ответы):
   - СКАЖИ ТОЛЬКО ЭТО: "[TZ_COMPLETE]"
   - и сразу после этого выведи ТЗ в формате JSON

3. ФОРМАТ ФИНАЛЬНОГО ТЗ (структура JSON):
{
  "project_overview": {
    "project_name": "Название проекта",
    "description": "Описание проекта",
    "goals": ["Цель 1", "Цель 2"],
    "target_audience": "Описание ЦА"
  },
  "requirements": {
    "functional": [
      {
        "id": "FR-001",
        "description": "Описание функционального требования",
        "priority": "высокий/средний/низкий"
      }
    ],
    "non_functional": [
      {
        "id": "NFR-001",
        "description": "Описание нефункционального требования",
        "category": "производительность/безопасность/удобство"
      }
    ]
  },
  "constraints": {
    "technical": ["Ограничение 1", "Ограничение 2"],
    "business": ["Бизнес-ограничение 1"],
    "assumptions": ["Допущение 1"]
  },
  "success_criteria": {
    "kpis": ["KPI 1", "KPI 2"],
    "acceptance_criteria": ["Критерий приемки 1"]
  },
  "metadata": {
    "estimated_timeline": "Оценка сроков",
    "budget_estimate": "Оценка бюджета",
    "risks": ["Риск 1", "Риск 2"]
  }
}

ВНИМАНИЕ:
- Не форсируй завершение. Задавай достаточно вопросов.
- [TZ_COMPLETE] - это специальный маркер, который говорит, что информация собрана.
- После [TZ_COMPLETE] сразу выводи ТЗ в JSON без дополнительного текста.
- Если информации недостаточно, продолжай задавать вопросы."""
}


def init_session():
    """Инициализация сессии с настройками по умолчанию"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'chat_mode' not in session:
        session['chat_mode'] = 'default'  # default, json, tz_collection
    if 'tz_data' not in session:
        session['tz_data'] = None
    if 'tz_complete' not in session:
        session['tz_complete'] = False
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


@app.route('/')
def index():
    """Главная страница с чатом"""
    init_session()
    return render_template('index.html',
                           history=session['chat_history'],
                           chat_mode=session['chat_mode'],
                           tz_complete=session['tz_complete'])


@app.route('/send_message', methods=['POST'])
def send_message():
    """Обработка отправки сообщения"""
    try:
        logger.info(f"Получен запрос: {request.json}")
        user_message = request.json.get('message', '').strip()
        require_json = request.json.get('require_json', False)

        if not user_message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400

        init_session()

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
        # Если пользователь запросил JSON, но не в режиме ТЗ, используем JSON режим
        if require_json and chat_mode != 'tz_collection':
            actual_mode = 'json_format'
        else:
            actual_mode = chat_mode

        # Получаем историю диалога из сессии
        chat_history = session.get('chat_history', [])

        # Добавляем сообщение пользователя в историю
        chat_history.append({'role': 'user', 'text': user_message})

        # Выбираем системный промпт в зависимости от режима
        system_prompt = SYSTEM_PROMPTS.get(actual_mode, SYSTEM_PROMPTS['default'])

        # Подготавливаем сообщения для API
        messages = [{'role': 'system', 'text': system_prompt}]

        # Добавляем историю диалога
        max_messages = 20 if actual_mode == 'tz_collection' else 15
        for msg in chat_history[-max_messages:]:
            messages.append({
                'role': msg['role'],
                'text': msg['text']
            })

        # Формируем запрос к YandexGPT API
        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json'
        }

        # Настраиваем параметры в зависимости от режима
        temperature = 0.3 if actual_mode == 'json_format' else 0.4
        max_tokens = 1500 if actual_mode == 'tz_collection' else 2000

        payload = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': temperature,
                'maxTokens': max_tokens
            },
            'messages': messages
        }

        # Логируем запрос
        logger.info(f"Chat mode: {actual_mode}, Require JSON: {require_json}")

        # Отправляем запрос к YandexGPT
        response = requests.post(YANDEX_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            logger.error(f"API Error {response.status_code}: {response.text}")
            return jsonify({'error': f'Ошибка API: {response.status_code}'}), 500

        # Парсим ответ
        result = response.json()

        # Получаем ответ ассистента
        if 'result' in result:
            assistant_message = result['result']['alternatives'][0]['message']['text']
        elif 'alternatives' in result:
            assistant_message = result['alternatives'][0]['message']['text']
        else:
            assistant_message = extract_message_from_response(result)

        # Обработка в зависимости от режима
        parsed_response = None
        final_assistant_message = assistant_message

        if actual_mode == 'tz_collection':
            # Проверяем, содержит ли ответ маркер завершения ТЗ
            if '[TZ_COMPLETE]' in assistant_message:
                # Извлекаем JSON после маркера
                json_start = assistant_message.find('[TZ_COMPLETE]') + len('[TZ_COMPLETE]')
                json_text = assistant_message[json_start:].strip()

                # Пытаемся распарсить JSON
                try:
                    json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        tz_data = json.loads(json_str)

                        # Сохраняем собранные данные
                        session['tz_data'] = tz_data
                        session['tz_complete'] = True

                        # Формируем финальное сообщение
                        final_assistant_message = f'✅ Техническое Задание сформировано!\n\n```json\n{json.dumps(tz_data, ensure_ascii=False, indent=2)}\n```'

                        # Добавляем в историю только финальное сообщение
                        chat_history = [{
                            'role': 'assistant',
                            'text': final_assistant_message
                        }]

                        return jsonify({
                            'response': final_assistant_message,
                            'tz_complete': True,
                            'tz_data': tz_data,
                            'history': chat_history,
                            'chat_mode': chat_mode,
                            'require_json': require_json
                        })
                    else:
                        # Если JSON не найден, удаляем маркер и продолжаем
                        assistant_message = assistant_message.replace('[TZ_COMPLETE]', '')
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка парсинга JSON ТЗ: {str(e)}")
                    assistant_message = assistant_message.replace('[TZ_COMPLETE]', '')

        elif actual_mode == 'json_format':
            # Пытаемся распарсить JSON ответ
            try:
                json_match = re.search(r'\{.*\}', assistant_message, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)

                    # Проверяем обязательные поля
                    if 'response' not in parsed_response:
                        parsed_response['response'] = assistant_message

                    # Форматируем JSON для отображения
                    final_assistant_message = json.dumps(parsed_response, ensure_ascii=False, indent=2)
                else:
                    # Если JSON не найден, создаем структуру с исходным текстом
                    parsed_response = {
                        "response": assistant_message,
                        "sentiment": "нейтральный",
                        "confidence": 0.5,
                        "entities": [],
                        "intent": "unknown",
                        "suggestions": []
                    }
                    final_assistant_message = json.dumps(parsed_response, ensure_ascii=False, indent=2)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                parsed_response = {
                    "response": assistant_message,
                    "error": "Не удалось сгенерировать валидный JSON",
                    "sentiment": "нейтральный",
                    "confidence": 0.1,
                    "entities": [],
                    "intent": "unknown",
                    "suggestions": []
                }
                final_assistant_message = json.dumps(parsed_response, ensure_ascii=False, indent=2)

        # Добавляем ответ ассистента в историю
        chat_history.append({'role': 'assistant', 'text': final_assistant_message})

        # Ограничиваем историю (максимум 50 сообщений)
        if len(chat_history) > 50:
            chat_history = chat_history[-50:]

        # Сохраняем обновленную историю в сессии
        session['chat_history'] = chat_history
        session.modified = True

        # Формируем ответ для клиента
        response_data = {
            'response': final_assistant_message,
            'history': chat_history,
            'chat_mode': chat_mode,
            'require_json': require_json,
            'tz_complete': session.get('tz_complete', False)
        }

        if parsed_response:
            response_data['parsed'] = parsed_response
        if session.get('tz_data'):
            response_data['tz_data'] = session['tz_data']

        # Логируем полный ответ
        logger.info(f"Отправляем ответ: {json.dumps(response_data, ensure_ascii=False)}")

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


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """Установка режима работы чата"""
    try:
        mode = request.json.get('mode', 'default')

        if mode not in ['default', 'json_format', 'tz_collection']:
            return jsonify({'error': 'Неверный режим'}), 400

        init_session()

        # Если переключаемся из режима ТЗ, сбрасываем состояние
        if mode != 'tz_collection':
            session['tz_complete'] = False
            session['tz_data'] = None

        # Если включаем режим ТЗ, инициализируем историю
        if mode == 'tz_collection':
            session['chat_history'] = [{
                'role': 'assistant',
                'text': 'Здравствуйте! Я помогу вам составить Техническое Задание. Расскажите, пожалуйста, о вашем проекте.'
            }]

        session['chat_mode'] = mode
        session.modified = True

        return jsonify({
            'success': True,
            'mode': mode,
            'message': f'Режим изменен на: {mode}'
        })
    except Exception as e:
        logger.error(f"Error setting mode: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Очистка истории диалога"""
    session['chat_history'] = []
    session['tz_data'] = None
    session['tz_complete'] = False
    session.modified = True
    return jsonify({'success': True})


@app.route('/api_info')
def api_info():
    """Информация о статусе API"""
    status = 'configured' if YANDEX_API_KEY and YANDEX_FOLDER_ID else 'not_configured'
    api_key_preview = YANDEX_API_KEY[:10] + '...' if YANDEX_API_KEY else 'не установлен'
    folder_id_preview = YANDEX_FOLDER_ID[:10] + '...' if YANDEX_FOLDER_ID else 'не установлен'

    return jsonify({
        'status': status,
        'has_api_key': bool(YANDEX_API_KEY),
        'has_folder_id': bool(YANDEX_FOLDER_ID),
        'api_key_preview': api_key_preview,
        'folder_id_preview': folder_id_preview
    })


@app.route('/test_api', methods=['GET'])
def test_api():
    """Тестовый запрос к API для проверки подключения"""
    try:
        if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
            return jsonify({'error': 'API ключ или Folder ID не установлены'}), 400

        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json'
        }

        test_payload = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': 0.1,
                'maxTokens': 100
            },
            'messages': [
                {
                    'role': 'system',
                    'text': SYSTEM_PROMPTS['default']
                },
                {
                    'role': 'user',
                    'text': 'Привет!'
                }
            ]
        }

        response = requests.post(YANDEX_API_URL, headers=headers, json=test_payload, timeout=10)

        return jsonify({
            'status_code': response.status_code,
            'response_text': response.text[:500] if response.text else 'Пустой ответ'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_session_info', methods=['GET'])
def get_session_info():
    """Получить информацию о текущей сессии"""
    init_session()
    return jsonify({
        'chat_mode': session.get('chat_mode', 'default'),
        'tz_complete': session.get('tz_complete', False),
        'tz_data': session.get('tz_data'),
        'history_length': len(session.get('chat_history', [])),
        'session_id': session.get('session_id')
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)