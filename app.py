import os
import json
import requests
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from datetime import timedelta
import logging
import re

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


@app.route('/')
def index():
    """Главная страница с чатом"""
    # Инициализация истории диалога в сессии
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', history=session['chat_history'])


@app.route('/send_message', methods=['POST'])
def send_message():
    """Обработка отправки сообщения"""
    try:
        user_message = request.json.get('message', '').strip()
        require_json = request.json.get('require_json', False)

        if not user_message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400

        # Получаем историю диалога из сессии
        chat_history = session.get('chat_history', [])

        # Добавляем сообщение пользователя в историю
        chat_history.append({'role': 'user', 'text': user_message})

        # Определяем системный промпт в зависимости от формата ответа
        if require_json:
            system_prompt = """Ты полезный ассистент. Всегда отвечай в формате JSON со следующей структурой:
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
            
            Не добавляй никаких дополнительных комментариев кроме JSON!"""
        else:
            system_prompt = 'Ты полезный ассистент. Отвечай вежливо и по существу.'

        # Подготавливаем системный промпт и сообщения для API
        # YandexGPT ожидает структуру с полем "text" а не "content"
        messages = [
            {
                'role': 'system',
                'text': system_prompt
            }
        ]

        # Добавляем историю диалога (последние 10 сообщений)
        # Преобразуем нашу историю в формат, ожидаемый YandexGPT
        for msg in chat_history[-10:]:
            # В YandexGPT roles: 'user', 'assistant', 'system'
            role = msg['role']
            messages.append({
                'role': role,
                'text': msg['text']
            })

        # Формируем запрос к YandexGPT API согласно документации
        headers = {
            'Authorization': f'Api-Key {YANDEX_API_KEY}',
            'Content-Type': 'application/json'
        }

        # Согласно документации YandexGPT, модель должна быть указана как yandexgpt/latest
        payload = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': 0.3,  # Уменьшил температуру для более стабильного JSON
                'maxTokens': 2000
            },
            'messages': messages
        }

        # Логируем запрос (без ключа)
        logger.info(f"Sending request to YandexGPT API with modelUri: {payload['modelUri']}")
        logger.info(f"Sending request to YandexGPT API with require_json={require_json}")
        logger.info(f"Messages count: {len(messages)}")

        # Отправляем запрос к YandexGPT
        response = requests.post(YANDEX_API_URL, headers=headers, json=payload, timeout=30)

        # Логируем статус ответа
        logger.info(f"API Response Status: {response.status_code}")

        # Проверяем статус ответа
        if response.status_code != 200:
            logger.error(f"API Error {response.status_code}: {response.text}")
            # Пробуем другой формат модели, если первый не работает
            return try_alternative_model(user_message, chat_history, headers)

        # Парсим ответ
        result = response.json()
        logger.info(f"API Response: {json.dumps(result, ensure_ascii=False)[:200]}...")

        # Получаем ответ ассистента
        # Структура ответа может отличаться, проверяем несколько вариантов
        if 'result' in result:
            assistant_message = result['result']['alternatives'][0]['message']['text']
        elif 'alternatives' in result:
            assistant_message = result['alternatives'][0]['message']['text']
        else:
            # Если структура неожиданная, ищем текст ответа
            assistant_message = extract_message_from_response(result)

        # Если требуется JSON, пытаемся распарсить
        parsed_response = None
        if require_json:
            try:
                # Пытаемся найти JSON в ответе (модель может добавить текст до/после JSON)
                json_match = re.search(r'\{.*}', assistant_message, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_response = json.loads(json_str)

                    # Проверяем обязательные поля
                    if 'response' not in parsed_response:
                        parsed_response['response'] = assistant_message

                    # Сохраняем полный ответ в историю
                    assistant_message = json.dumps(parsed_response, ensure_ascii=False)
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
                    assistant_message = json.dumps(parsed_response, ensure_ascii=False)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                # Если не удалось распарсить, создаем JSON с ошибкой
                parsed_response = {
                    "response": assistant_message,
                    "error": "Не удалось сгенерировать валидный JSON",
                    "sentiment": "нейтральный",
                    "confidence": 0.1,
                    "entities": [],
                    "intent": "unknown",
                    "suggestions": []
                }
                assistant_message = json.dumps(parsed_response, ensure_ascii=False)

        # Добавляем ответ ассистента в историю
        chat_history.append({'role': 'assistant', 'text': assistant_message})

        # Сохраняем обновленную историю в сессии
        session['chat_history'] = chat_history
        session.modified = True

        # Формируем ответ для клиента
        response_data = {
            'response': assistant_message,
            'history': chat_history,
            'require_json': require_json
        }

        if parsed_response and require_json:
            response_data['parsed'] = parsed_response

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({'error': 'Ошибка соединения с YandexGPT API. Проверьте API ключ и Folder ID.'}), 500
    except KeyError as e:
        logger.error(f"Key error in response parsing: {str(e)}")
        logger.error(f"Response was: {response.text if 'response' in locals() else 'No response'}")
        return jsonify({'error': 'Ошибка обработки ответа от YandexGPT. Проверьте формат запроса.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500


def try_alternative_model(user_message, chat_history, headers):
    """Попробовать альтернативный формат модели"""
    try:
        # Пробуем другой вариант URI модели
        alternative_payload = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite/latest',
            'completionOptions': {
                'stream': False,
                'temperature': 0.6,
                'maxTokens': 1000
            },
            'messages': [
                {
                    'role': 'system',
                    'text': 'Ты полезный ассистент. Отвечай кратко и по существу.'
                },
                {
                    'role': 'user',
                    'text': user_message
                }
            ]
        }

        logger.info("Trying alternative model: yandexgpt-lite/latest")
        response = requests.post(YANDEX_API_URL, headers=headers, json=alternative_payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            assistant_message = result['result']['alternatives'][0]['message']['text']

            # Обновляем историю
            chat_history.append({'role': 'user', 'text': user_message})
            chat_history.append({'role': 'assistant', 'text': assistant_message})

            return jsonify({
                'response': assistant_message,
                'history': chat_history,
                'note': 'Использована облегченная модель'
            })
        else:
            return jsonify({'error': f'API вернул ошибку {response.status_code}. Проверьте настройки API.'}), 500

    except Exception as e:
        logger.error(f"Alternative model also failed: {str(e)}")
        return jsonify({'error': 'Ошибка API. Проверьте правильность API ключа и Folder ID.'}), 500


def extract_message_from_response(result):
    """Извлечь сообщение из ответа API при нестандартной структуре"""
    try:
        # Пробуем разные возможные пути к тексту ответа
        if 'result' in result:
            if 'alternatives' in result['result']:
                return result['result']['alternatives'][0]['message']['text']

        # Ищем рекурсивно
        import json
        result_str = json.dumps(result)
        if '"text":' in result_str:
            # Простая эвристика для извлечения текста
            import re
            match = re.search(r'"text":\s*"([^"]+)"', result_str)
            if match:
                return match.group(1)

        return "Ответ получен, но не удалось распарсить структуру."
    except:
        return "Ошибка при обработке ответа от модели."


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Очистка истории диалога"""
    session['chat_history'] = []
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
                    'text': 'Ты тестовый ассистент. Ответь одним словом.'
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
            'response_text': response.text[:500] if response.text else 'Пустой ответ',
            'headers_sent': {'Authorization': 'Api-Key *****', 'Content-Type': 'application/json'},
            'payload_sent': test_payload
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
