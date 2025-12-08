import json
import logging
import os
import re
import uuid
from datetime import timedelta

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session

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
        session['temperature'] = "0.7"  # Сохраняем как строку для совместимости с select


@app.route('/')
def index():
    """Главная страница с чатом"""
    init_session()
    return render_template('index.html',
                           history=session['chat_history'],
                           chat_mode=session['chat_mode'],
                           tz_complete=session['tz_complete'],
                           temperature=session['temperature'],
                           current_role=session.get('current_role_name', 'Обычный ассистент'))


@app.route('/send_message', methods=['POST'])
def send_message():
    """Обработка отправки сообщения"""
    try:
        logger.info(f"Получен запрос: {request.json}")
        user_message = request.json.get('message', '').strip()
        require_json = request.json.get('require_json', False)

        # Получаем температуру из запроса
        temperature = request.json.get('temperature')
        if temperature is not None:
            try:
                # Преобразуем в float и ограничиваем в допустимых пределах YandexGPT API (0-1.0)
                temperature_val = float(temperature)
                temperature_val = max(0.0, min(1.0, temperature_val))  # Максимум 1.0 для YandexGPT
            except (ValueError, TypeError):
                temperature_val = session.get('temperature', 0.7)
        else:
            temperature_val = session.get('temperature', 0.7)

        if not user_message:
            return jsonify({'error': 'Сообщение не может быть пустым'}), 400

        init_session()

        # Сохраняем температуру в сессии (сохраняем исходное значение, а не ограниченное)
        session['temperature'] = temperature_val

        # Проверяем, является ли сообщение командой для смены роли
        if user_message.lower().startswith('/role '):
            return handle_role_command(user_message)

        # Проверяем, является ли сообщение командой для установки кастомного промпта
        elif user_message.lower().startswith('/system '):
            return handle_system_command(user_message)

        # Проверяем, является ли сообщение командой для показа текущей роли
        elif user_message.lower() in ['/role', '/current', '/whoami']:
            return show_current_role()

        # Проверяем, является ли сообщение командой для сброса роли
        elif user_message.lower() in ['/reset', '/default', '/clearrole']:
            return reset_role()

        # Проверяем, является ли сообщение командой помощи
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

        # Добавляем сообщение пользователя в историю (только user сообщения)
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

        # Добавляем историю диалога (фильтруем только user и assistant сообщения)
        max_messages = 20 if actual_mode == 'tz_collection' else 15
        # Фильтруем только сообщения пользователя и ассистента (игнорируем system)
        filtered_history = []
        for msg in chat_history:
            if msg['role'] in ['user', 'assistant']:
                filtered_history.append(msg)

        # Берем только последние сообщения для контекста
        for msg in filtered_history[-max_messages:]:
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
        # Важно: не изменяем температуру, выбранную пользователем, если она не равна 0
        if actual_mode == 'json_format':
            # Для JSON формата используем низкую температуру для более структурированных ответов
            # Но если пользователь выбрал 0, оставляем 0
            if temperature_val > 0.3:
                temperature_val = 0.3
            max_tokens = 1500
        elif actual_mode == 'tz_collection':
            # Для сбора ТЗ умеренная температура
            # Но если пользователь выбрал 0, оставляем 0
            if temperature_val > 0.4:
                temperature_val = 0.4
            max_tokens = 2000
        elif session.get('custom_system_prompt'):
            # Для кастомных ролей используем умеренную температуру
            max_tokens = 2000
        else:
            max_tokens = 1500

        payload = {
            'modelUri': f'gpt://{YANDEX_FOLDER_ID}/yandexgpt/latest',
            'completionOptions': {
                'stream': False,
                'temperature': temperature_val,
                'maxTokens': max_tokens
            },
            'messages': messages
        }

        # Логируем запрос
        logger.info(f"Chat mode: {actual_mode}, Temperature: {temperature_val}, Messages in request: {len(messages)}")

        # Отправляем запрос к YandexGPT
        response = requests.post(YANDEX_API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            logger.error(f"API Error {response.status_code}: {response.text}")
            # Пробуем получить больше информации об ошибке
            try:
                error_details = response.json()
                logger.error(f"API Error details: {error_details}")
            except:
                pass
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
                            'temperature': temperature_val,
                            'require_json': require_json,
                            'current_role': session.get('current_role_name', 'Обычный ассистент')
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
            'history_count': len(chat_history)
        }

        if parsed_response:
            response_data['parsed'] = parsed_response
        if session.get('tz_data'):
            response_data['tz_data'] = session['tz_data']

        # Логируем полный ответ
        logger.info(
            f"Отправляем ответ: history_count={len(chat_history)}, temperature={temperature_val}, current_role={session.get('current_role_name')}")

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
        # Извлекаем название роли из команды
        role_text = user_message[6:].strip()  # Убираем "/role "

        # Предопределенные роли с их промптами
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
            },

            'ученый': {
                'prompt': 'Ты — педантичный ученый-исследователь. Твои ответы строго научны, содержат ссылки на исследования, статистику и факты. Ты избегаешь субъективных оценок, оперируешь только проверенными данными. Всегда указываешь степень достоверности информации и возможные погрешности.',
                'name': 'Ученый'
            },

            'философ': {
                'prompt': 'Ты — глубокий философ, размышляющий о фундаментальных вопросах бытия. Твои ответы содержат много вопросов, парадоксов и ссылок на философские учения. Ты видишь проблему с разных сторон и не даешь однозначных ответов, а приглашаешь к размышлению.',
                'name': 'Философ'
            },

            'юморист': {
                'prompt': 'Ты — остроумный комик, который находит смешное в любой ситуации. Твои ответы полны шуток, иронии и сарказма. Даже на серьезные вопросы ты отвечаешь с юмором, но при этом можешь донести важную мысль через призму комедии.',
                'name': 'Юморист'
            },

            'детектив': {
                'prompt': 'Ты — проницательный детектив в стиле нуар. Ты видишь скрытые мотивы, находишь связи между, казалось бы, несвязанными фактами. Твои ответы полны подозрений, вопросов и поиска истины. Ты всегда скептически относишься к поверхностным объяснениям.',
                'name': 'Детектив'
            }
        }

        # Ищем совпадение
        matched_role = None
        role_name = None

        for key, value in predefined_roles.items():
            if key.startswith(role_text.lower()) or role_text.lower() in key:
                matched_role = value
                role_name = value['name']
                break

        if not matched_role:
            # Если роль не найдена, возвращаем список доступных ролей
            available_roles = ", ".join(predefined_roles.keys())
            return jsonify({
                'response': f"❌ Роль '{role_text}' не найдена. Доступные роли: {available_roles}\n\nИспользуйте команду: /role [название_роли]",
                'is_command': True,
                'current_role': session.get('current_role_name', 'Обычный ассистент'),
                'history_count': len(session.get('chat_history', []))
            })

        # Устанавливаем кастомный промпт
        session['custom_system_prompt'] = matched_role['prompt']
        session['chat_mode'] = 'custom'
        session['current_role_name'] = role_name

        # Добавляем системное сообщение в историю (только для отображения, не для API)
        chat_history = session.get('chat_history', [])
        system_message = f"🔄 Роль изменена на: **{role_name}**\n\nТеперь я буду отвечать как {role_name}."
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
        # Извлекаем промпт из команды
        system_prompt = user_message[8:].strip()  # Убираем "/system "

        if not system_prompt:
            return jsonify({
                'response': "❌ Укажите системный промпт. Например: /system Ты теперь кот, который говорит мяу.",
                'is_command': True,
                'current_role': session.get('current_role_name', 'Обычный ассистент'),
                'history_count': len(session.get('chat_history', []))
            })

        # Устанавливаем кастомный промпт
        session['custom_system_prompt'] = system_prompt
        session['chat_mode'] = 'custom'

        # Создаем короткое название роли из промпта
        role_name = "Кастомная роль"
        if len(system_prompt) > 30:
            role_name = system_prompt[:30] + "..."
        else:
            role_name = system_prompt

        session['current_role_name'] = role_name

        # Добавляем системное сообщение в историю
        chat_history = session.get('chat_history', [])
        system_message = f"🔄 Установлен кастомный системный промпт:\n\n**{system_prompt[:200]}...**" if len(
            system_prompt) > 200 else f"🔄 Установлен кастомный системный промпт:\n\n**{system_prompt}**"
        chat_history.append({'role': 'system', 'text': system_message})
        session['chat_history'] = chat_history
        session.modified = True

        return jsonify({
            'response': system_message,
            'history': chat_history,
            'is_command': True,
            'current_role': role_name,
            'history_count': len(chat_history),
            'prompt_preview': system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
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
    custom_prompt = session.get('custom_system_prompt')

    response_message = f"👤 **Текущая роль:** {current_role}\n\n"

    if custom_prompt:
        response_message += f"**Системный промпт:**\n{custom_prompt[:300]}..."
        if len(custom_prompt) > 300:
            response_message += f"\n\n_(показано 300 из {len(custom_prompt)} символов)_"
    else:
        response_message += f"**Режим:** {session.get('chat_mode', 'default')}\n"
        response_message += f"**Системный промпт:** {SYSTEM_PROMPTS.get(session.get('chat_mode', 'default'), 'Не указан')[:300]}..."

    return jsonify({
        'response': response_message,
        'is_command': True,
        'current_role': current_role,
        'history_count': len(session.get('chat_history', []))
    })


def reset_role():
    """Сбросить роль к значениям по умолчанию"""
    # Сбрасываем кастомный промпт
    session['custom_system_prompt'] = None
    session['chat_mode'] = 'default'
    session['current_role_name'] = 'Обычный ассистент'

    # Добавляем системное сообщение в историю
    chat_history = session.get('chat_history', [])
    system_message = "🔄 Роль сброшена к значениям по умолчанию. Теперь я обычный ассистент."
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

**Смена роли:**
• `/role [название]` - сменить роль (инженер, режиссер, бабушка, ученый, философ, юморист, детектив)
• `/system [промпт]` - установить произвольный системный промпт
• `/reset` - сбросить роль к значениям по умолчанию
• `/role` или `/current` - показать текущую роль

**Управление:**
• `/help` - показать эту справку
• `/clear` - очистить историю диалога

**Эксперимент:**
1. Задайте вопрос в текущей роли
2. Смените роль командой `/role инженер`
3. Задайте тот же вопрос или продолжайте диалог
4. Наблюдайте, как меняется стиль ответов!

**Пример:**
    /role инженер
    Как улучшить общение в цифровую эпоху?

    /role режиссер
    А теперь объясни то же самое по-другому!
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

        # Все допустимые режимы
        valid_modes = ['default', 'json_format', 'tz_collection']

        if mode not in valid_modes:
            return jsonify({'error': 'Неверный режим'}), 400

        init_session()

        # Записываем смену роли в историю
        old_mode = session.get('chat_mode', 'default')

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
        else:
            # Для других режимов сбрасываем кастомный промпт
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

    # Добавляем системное сообщение
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
    chat_history = session.get('chat_history', [])

    return jsonify({
        'chat_mode': session.get('chat_mode', 'default'),
        'temperature': session.get('temperature', 0.7),
        'tz_complete': session.get('tz_complete', False),
        'tz_data': session.get('tz_data'),
        'history_length': len(chat_history),
        'session_id': session.get('session_id'),
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'has_custom_prompt': bool(session.get('custom_system_prompt'))
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)