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
        'cost_per_1k_tokens': 0.0003,
        'max_tokens': 4000,
        'description': 'Базовая модель Yandex'
    },
    'yandexgpt': {
        'name': 'YandexGPT Pro',
        'model_uri': 'yandexgpt/latest',
        'cost_per_1k_tokens': 0.0012,
        'max_tokens': 8000,
        'description': 'Продвинутая модель Yandex'
    },
    # 'qwen': {
    #     'name': 'Qwen 2.5 32B',
    #     'model_uri': 'qwen/qwen-2.5-32b-instruct',
    #     'cost_per_1k_tokens': 0.0008,
    #     'max_tokens': 32000,
    #     'description': 'Мощная модель от Alibaba'
    # },
    # 'gemma': {
    #     'name': 'Gemma 2 9B',
    #     'model_uri': 'gemma/gemma-2-9b-it',
    #     'cost_per_1k_tokens': 0.0005,
    #     'max_tokens': 16000,
    #     'description': 'Эффективная модель от Google'
    # }
}

# Базовые системные промпты для разных режимов
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


def get_model_config(model_key):
    """Получить конфигурацию модели по ключу"""
    return MODELS_CONFIG.get(model_key, MODELS_CONFIG['yandexgpt-lite'])


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
            'temperature': temperature,
            'maxTokens': max_tokens
        },
        'messages': messages
    }

    # Замер времени выполнения
    start_time = time.time()
    response = requests.post(YANDEX_API_BASE_URL, headers=headers, json=payload, timeout=60)
    end_time = time.time()
    execution_time = end_time - start_time

    if response.status_code != 200:
        logger.error(f"API Error {response.status_code}: {response.text}")
        return None, execution_time, 0, 0.0

    result = response.json()

    # Получаем количество токенов из ответа
    total_tokens = 0
    try:
        if 'result' in result and 'usage' in result['result']:
            total_tokens = int(result['result']['usage'].get('totalTokens', 0))
        elif 'usage' in result:
            total_tokens = int(result['usage'].get('totalTokens', 0))
    except (ValueError, TypeError, KeyError) as e:
        logger.error(f"Ошибка при получении количества токенов: {e}")
        total_tokens = 0

    # Рассчитываем стоимость
    try:
        cost = (total_tokens / 1000.0) * float(model_config['cost_per_1k_tokens'])
    except (TypeError, ValueError) as e:
        logger.error(f"Ошибка при расчете стоимости: {e}")
        cost = 0.0

    return result, execution_time, total_tokens, cost


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
                           selected_model=session.get('selected_model', 'yandexgpt-lite'))


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

        # Получаем историю диалога из сессии
        chat_history = session.get('chat_history', [])

        # Добавляем сообщение пользователя в историю
        chat_history.append({'role': 'user', 'text': user_message})

        # Выбираем системный промпт в зависимости от режима
        if session.get('custom_system_prompt'):
            system_prompt = session['custom_system_prompt']
        else:
            system_prompt = SYSTEM_PROMPTS.get(actual_mode, SYSTEM_PROMPTS['default'])

        # Сохраняем текущий промпт для будущих запросов
        session['last_system_prompt'] = system_prompt

        # Подготавливаем сообщения для API
        messages = [{'role': 'system', 'text': system_prompt}]

        # Добавляем историю диалога
        max_messages = 20 if actual_mode == 'tz_collection' else 15
        filtered_history = []
        for msg in chat_history:
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
            max_tokens = min(1500, model_config['max_tokens'])
        elif actual_mode == 'tz_collection':
            max_tokens = min(2000, model_config['max_tokens'])
        elif session.get('custom_system_prompt'):
            max_tokens = min(2000, model_config['max_tokens'])
        else:
            max_tokens = min(1500, model_config['max_tokens'])

        # Вызываем API с замером метрик
        result, execution_time, total_tokens, cost = call_yandex_api(
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
                            'temperature': temperature_val,
                            'require_json': require_json,
                            'current_role': session.get('current_role_name', 'Обычный ассистент'),
                            'model': model_key,
                            'model_name': model_config['name'],
                            'execution_time': round(execution_time, 2),
                            'total_tokens': total_tokens,
                            'cost': round(cost, 5)
                        })
                    else:
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
                    "sentiment": "neutral",
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
            'temperature': temperature_val,
            'require_json': require_json,
            'tz_complete': session.get('tz_complete', False),
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(chat_history),
            'model': model_key,
            'model_name': model_config['name'],
            'execution_time': round(execution_time, 2),
            'total_tokens': total_tokens,
            'cost': round(cost, 5)
        }

        if parsed_response:
            response_data['parsed'] = parsed_response
        if session.get('tz_data'):
            response_data['tz_data'] = session['tz_data']

        logger.info(
            f"Отправляем ответ: model={model_key}, tokens={total_tokens}, time={execution_time:.2f}s, cost={cost:.5f} руб")

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
                'prompt': 'Ты — инженер-прагматик с 20-летним стажем. Ты ненавидишь расплывчатые формулировки, требуешь точности и конкретики. Ты веришь только данным, логике и проверенным решениям. Твои ответы структурированы, ты любишь списки "за" и "против". Ты всегда ищешь подвох и скрытые риски в любой идее. Твой девиз: "Если что-то работает, не трогай это. Если не работает — найди спецификацию".',
                'name': 'Инженер'
            },
            'режиссер': {
                'prompt': 'Ты — знаменитый режиссёр с безграничной фантазией. Ты видишь мир через призму кино, метафор и архетипов. Твои ответы полны визуальных образов, ты мыслишь историями и персонажами. Ты обожаешь гиперболу, драматизацию и неожиданные повороты. Технические детали для тебя лишь фон для большой человеческой драмы. Твой главный вопрос всегда: "А где здесь конфликт и эмоция?".',
                'name': 'Режиссер'
            },
            'бабушка': {
                'prompt': 'Ты — добрая, мудрая бабушка, которая повидала многое на своём веку. Ты говоришь просто, с теплотой и лёгкой грустью. Ты веришь в народную мудрость, интуицию и важность простых человеческих ценностей: семья, покой, доброта. Ты любишь вспоминать аналогии из жизни, давать утешительные и практические советы. Технологии ты оцениваешь с точки зрения того, делают ли они людей счастливее. В твоих ответах всегда есть лёгкий налёт ностальгии.',
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


def show_help():
    """Показать справку по командам"""
    help_text = """
📚 **Доступные команды:**

/role [инженер|режиссер|бабушка] - сменить роль
/system [промпт] - установить произвольный промпт
/reset - сбросить роль
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
    session['tz_data'] = None
    session['tz_complete'] = False
    session.modified = True

    system_message = "🗑️ История диалога очищена."

    return jsonify({
        'success': True,
        'response': system_message,
        'history': [{'role': 'system', 'text': system_message}],
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'history_count': 0
    })


@app.route('/api_info')
def api_info():
    """Информация о статусе API"""
    status = 'configured' if YANDEX_API_KEY and YANDEX_FOLDER_ID else 'not_configured'
    return jsonify({
        'status': status,
        'has_api_key': bool(YANDEX_API_KEY),
        'has_folder_id': bool(YANDEX_FOLDER_ID)
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
        'session_id': session.get('session_id'),
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'has_custom_prompt': bool(session.get('custom_system_prompt')),
        'selected_model': session.get('selected_model', 'yandexgpt-lite')
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