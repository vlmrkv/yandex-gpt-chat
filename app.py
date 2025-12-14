import json
import logging
import os
import re
import uuid
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session, send_from_directory, Response

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
}

# Конфигурация компрессии
COMPRESSION_CONFIG = {
    'enable_compression': True,
    'compression_interval': 10,
    'summary_model': 'yandexgpt-lite',
    'max_summary_tokens': 300,
    'keep_last_messages': 3,
    'min_compression_saving': 0.3,
}

# Базовые системные промпты
SYSTEM_PROMPTS = {
    'default': 'Ты полезный ассистент. Отвечай вежливо и по существу.',
    'json_format': 'Ты полезный ассистент. Всегда отвечай в формате JSON.',
    'tz_collection': 'Ты - профессиональный аналитик, который собирает требования для Технического Задания.',
    'history_compression': """Ты - эксперт по сжатию информации. Твоя задача - создать краткое изложение диалога.

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


# Класс для работы с внешней памятью (SQLite)
class MemoryStorage:
    """Класс для хранения промежуточных результатов в SQLite"""

    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Инициализация базы данных и создание таблиц"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Таблица для хранения сессий
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS sessions
                               (
                                   session_id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   created_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   updated_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   chat_mode
                                   TEXT
                                   DEFAULT
                                   'default',
                                   temperature
                                   REAL
                                   DEFAULT
                                   0.7,
                                   selected_model
                                   TEXT
                                   DEFAULT
                                   'yandexgpt-lite',
                                   custom_system_prompt
                                   TEXT,
                                   current_role_name
                                   TEXT
                                   DEFAULT
                                   'Обычный ассистент',
                                   metadata
                                   TEXT
                               )
                               ''')

                # Таблица для хранения сообщений
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS messages
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   session_id
                                   TEXT,
                                   message_index
                                   INTEGER,
                                   role
                                   TEXT,
                                   content
                                   TEXT,
                                   tokens
                                   INTEGER
                                   DEFAULT
                                   0,
                                   cost
                                   REAL
                                   DEFAULT
                                   0.0,
                                   model
                                   TEXT,
                                   temperature
                                   REAL,
                                   created_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   FOREIGN
                                   KEY
                               (
                                   session_id
                               ) REFERENCES sessions
                               (
                                   session_id
                               ) ON DELETE CASCADE,
                                   UNIQUE
                               (
                                   session_id,
                                   message_index
                               )
                                   )
                               ''')

                # Таблица для сжатой истории
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS compressed_history
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   session_id
                                   TEXT,
                                   summary
                                   TEXT,
                                   original_messages_count
                                   INTEGER,
                                   original_tokens
                                   INTEGER,
                                   compressed_tokens
                                   INTEGER,
                                   created_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   FOREIGN
                                   KEY
                               (
                                   session_id
                               ) REFERENCES sessions
                               (
                                   session_id
                               ) ON DELETE CASCADE
                                   )
                               ''')

                # Таблица для статистики токенов
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS token_stats
                               (
                                   session_id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   total_input_tokens
                                   INTEGER
                                   DEFAULT
                                   0,
                                   total_output_tokens
                                   INTEGER
                                   DEFAULT
                                   0,
                                   total_cost
                                   REAL
                                   DEFAULT
                                   0.0,
                                   updated_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   FOREIGN
                                   KEY
                               (
                                   session_id
                               ) REFERENCES sessions
                               (
                                   session_id
                               ) ON DELETE CASCADE
                                   )
                               ''')

                # Таблица для статистики компрессии
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS compression_stats
                               (
                                   session_id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   total_compressions
                                   INTEGER
                                   DEFAULT
                                   0,
                                   tokens_saved
                                   INTEGER
                                   DEFAULT
                                   0,
                                   original_tokens
                                   INTEGER
                                   DEFAULT
                                   0,
                                   compressed_tokens
                                   INTEGER
                                   DEFAULT
                                   0,
                                   compression_ratio
                                   REAL
                                   DEFAULT
                                   0.0,
                                   updated_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   FOREIGN
                                   KEY
                               (
                                   session_id
                               ) REFERENCES sessions
                               (
                                   session_id
                               ) ON DELETE CASCADE
                                   )
                               ''')

                # Таблица для промежуточных результатов
                cursor.execute('''
                               CREATE TABLE IF NOT EXISTS intermediate_results
                               (
                                   id
                                   INTEGER
                                   PRIMARY
                                   KEY
                                   AUTOINCREMENT,
                                   session_id
                                   TEXT,
                                   result_type
                                   TEXT,
                                   data
                                   TEXT,
                                   created_at
                                   TIMESTAMP
                                   DEFAULT
                                   CURRENT_TIMESTAMP,
                                   metadata
                                   TEXT,
                                   FOREIGN
                                   KEY
                               (
                                   session_id
                               ) REFERENCES sessions
                               (
                                   session_id
                               ) ON DELETE CASCADE
                                   )
                               ''')

                # Индексы для быстрого поиска
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_messages_session_index ON messages(session_id, message_index)')
                cursor.execute(
                    'CREATE INDEX IF NOT EXISTS idx_intermediate_session ON intermediate_results(session_id)')

                conn.commit()
                logger.info(f"База данных инициализирована: {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Ошибка инициализации базы данных: {e}")
            raise

    def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """Сохранить или обновить данные сессии"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                exists = cursor.fetchone()

                metadata = json.dumps({
                    'tz_data': session_data.get('tz_data'),
                    'tz_complete': session_data.get('tz_complete', False),
                    'last_system_prompt': session_data.get('last_system_prompt'),
                    'compression_enabled': session_data.get('compression_enabled', True)
                })

                if exists:
                    cursor.execute('''
                                   UPDATE sessions
                                   SET updated_at           = CURRENT_TIMESTAMP,
                                       chat_mode            = ?,
                                       temperature          = ?,
                                       selected_model       = ?,
                                       custom_system_prompt = ?,
                                       current_role_name    = ?,
                                       metadata             = ?
                                   WHERE session_id = ?
                                   ''', (
                                       session_data.get('chat_mode', 'default'),
                                       float(session_data.get('temperature', 0.7)),
                                       session_data.get('selected_model', 'yandexgpt-lite'),
                                       session_data.get('custom_system_prompt'),
                                       session_data.get('current_role_name', 'Обычный ассистент'),
                                       metadata,
                                       session_id
                                   ))
                else:
                    cursor.execute('''
                                   INSERT INTO sessions (session_id, chat_mode, temperature, selected_model,
                                                         custom_system_prompt, current_role_name, metadata)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)
                                   ''', (
                                       session_id,
                                       session_data.get('chat_mode', 'default'),
                                       float(session_data.get('temperature', 0.7)),
                                       session_data.get('selected_model', 'yandexgpt-lite'),
                                       session_data.get('custom_system_prompt'),
                                       session_data.get('current_role_name', 'Обычный ассистент'),
                                       metadata
                                   ))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения сессии: {e}")

    def save_message(self, session_id: str, message_index: int, role: str, content: str,
                     tokens: int = 0, cost: float = 0.0, model: str = None, temperature: float = None):
        """Сохранить сообщение в базу данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT OR REPLACE INTO messages 
                    (session_id, message_index, role, content, tokens, cost, model, temperature)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id, message_index, role, content,
                    tokens, cost, model, temperature
                ))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения сообщения: {e}")

    def save_compressed_history(self, session_id: str, summary: str,
                                original_messages_count: int, original_tokens: int,
                                compressed_tokens: int):
        """Сохранить сжатую историю"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                               INSERT INTO compressed_history
                               (session_id, summary, original_messages_count, original_tokens, compressed_tokens)
                               VALUES (?, ?, ?, ?, ?)
                               ''', (
                                   session_id, summary, original_messages_count,
                                   original_tokens, compressed_tokens
                               ))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения сжатой истории: {e}")

    def update_token_stats(self, session_id: str, input_tokens: int = 0,
                           output_tokens: int = 0, cost: float = 0.0):
        """Обновить статистику токенов"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT session_id FROM token_stats WHERE session_id = ?", (session_id,))
                exists = cursor.fetchone()

                if exists:
                    cursor.execute('''
                                   UPDATE token_stats
                                   SET total_input_tokens  = total_input_tokens + ?,
                                       total_output_tokens = total_output_tokens + ?,
                                       total_cost          = total_cost + ?,
                                       updated_at          = CURRENT_TIMESTAMP
                                   WHERE session_id = ?
                                   ''', (input_tokens, output_tokens, cost, session_id))
                else:
                    cursor.execute('''
                                   INSERT INTO token_stats
                                       (session_id, total_input_tokens, total_output_tokens, total_cost)
                                   VALUES (?, ?, ?, ?)
                                   ''', (session_id, input_tokens, output_tokens, cost))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Ошибка обновления статистики токенов: {e}")

    def update_compression_stats(self, session_id: str, stats: Dict[str, Any]):
        """Обновить статистику компрессии"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT session_id FROM compression_stats WHERE session_id = ?", (session_id,))
                exists = cursor.fetchone()

                if exists:
                    cursor.execute('''
                                   UPDATE compression_stats
                                   SET total_compressions = ?,
                                       tokens_saved       = ?,
                                       original_tokens    = ?,
                                       compressed_tokens  = ?,
                                       compression_ratio  = ?,
                                       updated_at         = CURRENT_TIMESTAMP
                                   WHERE session_id = ?
                                   ''', (
                                       stats.get('total_compressions', 0),
                                       stats.get('tokens_saved', 0),
                                       stats.get('original_tokens', 0),
                                       stats.get('compressed_tokens', 0),
                                       stats.get('compression_ratio', 0.0),
                                       session_id
                                   ))
                else:
                    cursor.execute('''
                                   INSERT INTO compression_stats
                                   (session_id, total_compressions, tokens_saved,
                                    original_tokens, compressed_tokens, compression_ratio)
                                   VALUES (?, ?, ?, ?, ?, ?)
                                   ''', (
                                       session_id,
                                       stats.get('total_compressions', 0),
                                       stats.get('tokens_saved', 0),
                                       stats.get('original_tokens', 0),
                                       stats.get('compressed_tokens', 0),
                                       stats.get('compression_ratio', 0.0)
                                   ))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Ошибка обновления статистики компрессии: {e}")

    def save_intermediate_result(self, session_id: str, result_type: str,
                                 data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Сохранить промежуточный результат"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                data_json = json.dumps(data, ensure_ascii=False)
                metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

                cursor.execute('''
                               INSERT INTO intermediate_results
                                   (session_id, result_type, data, metadata)
                               VALUES (?, ?, ?, ?)
                               ''', (session_id, result_type, data_json, metadata_json))

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения промежуточного результата: {e}")

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Загрузить все данные сессии из базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                session_row = cursor.fetchone()

                if not session_row:
                    return None

                session_data = dict(session_row)

                if session_data.get('metadata'):
                    try:
                        metadata = json.loads(session_data['metadata'])
                        session_data.update(metadata)
                    except json.JSONDecodeError:
                        pass

                cursor.execute('''
                               SELECT role, content, tokens, cost, model, temperature
                               FROM messages
                               WHERE session_id = ?
                               ORDER BY message_index
                               ''', (session_id,))

                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'role': row['role'],
                        'text': row['content']
                    })

                cursor.execute('''
                               SELECT summary
                               FROM compressed_history
                               WHERE session_id = ?
                               ORDER BY created_at
                               ''', (session_id,))

                compressed_history = [row['summary'] for row in cursor.fetchall()]

                cursor.execute("SELECT * FROM token_stats WHERE session_id = ?", (session_id,))
                token_stats_row = cursor.fetchone()

                cursor.execute("SELECT * FROM compression_stats WHERE session_id = ?", (session_id,))
                compression_stats_row = cursor.fetchone()

                result = {
                    'session_id': session_id,
                    'chat_mode': session_data.get('chat_mode', 'default'),
                    'temperature': session_data.get('temperature', 0.7),
                    'selected_model': session_data.get('selected_model', 'yandexgpt-lite'),
                    'custom_system_prompt': session_data.get('custom_system_prompt'),
                    'current_role_name': session_data.get('current_role_name', 'Обычный ассистент'),
                    'chat_history': messages,
                    'compressed_history': compressed_history,
                    'tz_data': session_data.get('tz_data'),
                    'tz_complete': session_data.get('tz_complete', False),
                    'last_system_prompt': session_data.get('last_system_prompt'),
                    'compression_enabled': session_data.get('compression_enabled', True)
                }

                if token_stats_row:
                    token_stats = dict(token_stats_row)
                    result['total_input_tokens'] = token_stats.get('total_input_tokens', 0)
                    result['total_output_tokens'] = token_stats.get('total_output_tokens', 0)
                    result['total_cost'] = token_stats.get('total_cost', 0.0)

                if compression_stats_row:
                    compression_stats = dict(compression_stats_row)
                    result['compression_stats'] = {
                        'total_compressions': compression_stats.get('total_compressions', 0),
                        'tokens_saved': compression_stats.get('tokens_saved', 0),
                        'original_tokens': compression_stats.get('original_tokens', 0),
                        'compressed_tokens': compression_stats.get('compressed_tokens', 0),
                        'compression_ratio': compression_stats.get('compression_ratio', 0.0)
                    }

                logger.info(f"Сессия загружена из базы данных: {session_id}, сообщений: {len(messages)}")
                return result

        except sqlite3.Error as e:
            logger.error(f"Ошибка загрузки сессии: {e}")
            return None

    def delete_session(self, session_id: str):
        """Удалить сессию и все связанные данные"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
                logger.info(f"Сессия удалена: {session_id}")

        except sqlite3.Error as e:
            logger.error(f"Ошибка удаления сессии: {e}")

    def get_session_list(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Получить список всех сессий"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute('''
                               SELECT s.session_id,
                                      s.created_at,
                                      s.updated_at,
                                      s.chat_mode,
                                      s.current_role_name,
                                      COUNT(m.id)                                                              as message_count,
                                      COALESCE(ts.total_input_tokens, 0) +
                                      COALESCE(ts.total_output_tokens, 0)                                      as total_tokens
                               FROM sessions s
                                        LEFT JOIN messages m ON s.session_id = m.session_id
                                        LEFT JOIN token_stats ts ON s.session_id = ts.session_id
                               GROUP BY s.session_id
                               ORDER BY s.updated_at DESC LIMIT ?
                               ''', (limit,))

                sessions = []
                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row['session_id'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'chat_mode': row['chat_mode'],
                        'current_role_name': row['current_role_name'],
                        'message_count': row['message_count'],
                        'total_tokens': row['total_tokens']
                    })

                return sessions

        except sqlite3.Error as e:
            logger.error(f"Ошибка получения списка сессий: {e}")
            return []

    def export_session(self, session_id: str, format: str = 'json') -> Optional[str]:
        """Экспортировать сессию в JSON"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return None

            export_data = {
                'session_data': session_data,
                'export_info': {
                    'export_date': datetime.now().isoformat(),
                    'format': format,
                    'database_version': '1.0'
                }
            }

            if format == 'json':
                return json.dumps(export_data, ensure_ascii=False, indent=2)
            else:
                return None

        except Exception as e:
            logger.error(f"Ошибка экспорта сессии: {e}")
            return None

    def cleanup_old_sessions(self, days_old: int = 30):
        """Очистить старые сессии (старше указанного количества дней)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                               DELETE
                               FROM sessions
                               WHERE julianday('now') - julianday(updated_at) > ?
                               ''', (days_old,))

                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Удалено старых сессий: {deleted_count}")
                return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Ошибка очистки старых сессий: {e}")
            return 0

    def get_database_stats(self) -> Dict[str, Any]:
        """Получить статистику базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                stats = {}

                cursor.execute("SELECT COUNT(*) FROM sessions")
                stats['total_sessions'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM messages")
                stats['total_messages'] = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM compressed_history")
                stats['total_compressed_histories'] = cursor.fetchone()[0]

                cursor.execute("SELECT SUM(total_input_tokens + total_output_tokens) FROM token_stats")
                total_tokens = cursor.fetchone()[0]
                stats['total_tokens'] = total_tokens if total_tokens else 0

                if os.path.exists(self.db_path):
                    stats['database_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)

                return stats

        except sqlite3.Error as e:
            logger.error(f"Ошибка получения статистики базы данных: {e}")
            return {}

    def get_last_session_id(self):
        """Получить ID последней сессии (самой свежей по updated_at)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                               SELECT session_id
                               FROM sessions
                               ORDER BY updated_at DESC LIMIT 1
                               ''')
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения последней сессии: {e}")
            return None


# Инициализация внешней памяти
try:
    memory = MemoryStorage()
    logger.info("✅ Внешняя память (SQLite) подключена")
except Exception as e:
    logger.error(f"❌ Ошибка инициализации внешней памяти: {e}")
    memory = None


def init_session():
    """Инициализация сессии с загрузкой последней сессии из БД"""
    session_id = session.get('session_id')

    # Если memory доступен и в сессии нет ID, пробуем загрузить последнюю сессию
    if memory and not session_id:
        last_session_id = memory.get_last_session_id()
        if last_session_id:
            saved_session = memory.load_session(last_session_id)
            if saved_session:
                logger.info(f"✅ Автоматически загружена последняя сессия: {last_session_id}")

                # Восстанавливаем все данные из сохраненной сессии
                for key, value in saved_session.items():
                    session[key] = value

                session['session_id'] = last_session_id

                # Обновляем время доступа к сессии
                memory.save_session(last_session_id, {
                    'chat_mode': session.get('chat_mode', 'default'),
                    'temperature': session.get('temperature', 0.7),
                    'selected_model': session.get('selected_model', 'yandexgpt-lite'),
                    'custom_system_prompt': session.get('custom_system_prompt'),
                    'current_role_name': session.get('current_role_name', 'Обычный ассистент'),
                    'tz_data': session.get('tz_data'),
                    'tz_complete': session.get('tz_complete', False),
                    'last_system_prompt': session.get('last_system_prompt'),
                    'compression_enabled': COMPRESSION_CONFIG['enable_compression']
                })

                # Восстанавливаем историю чата в интерфейсе
                if session.get('chat_history'):
                    logger.info(f"Восстановлено сообщений: {len(session['chat_history'])}")

                return

    # Если memory доступен и в сессии есть ID, пробуем загрузить эту сессию
    if memory and session_id:
        saved_session = memory.load_session(session_id)
        if saved_session:
            logger.info(f"✅ Сессия загружена из внешней памяти: {session_id}")

            for key, value in saved_session.items():
                session[key] = value

            session['session_id'] = session_id

            memory.save_session(session_id, {
                'chat_mode': session.get('chat_mode', 'default'),
                'temperature': session.get('temperature', 0.7),
                'selected_model': session.get('selected_model', 'yandexgpt-lite'),
                'custom_system_prompt': session.get('custom_system_prompt'),
                'current_role_name': session.get('current_role_name', 'Обычный ассистент'),
                'tz_data': session.get('tz_data'),
                'tz_complete': session.get('tz_complete', False),
                'last_system_prompt': session.get('last_system_prompt'),
                'compression_enabled': COMPRESSION_CONFIG['enable_compression']
            })

            return

    # Новая сессия (когда нет сохраненных сессий или memory недоступен)
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

    # Сохраняем новую сессию во внешней памяти
    if memory:
        memory.save_session(session['session_id'], {
            'chat_mode': session['chat_mode'],
            'temperature': session['temperature'],
            'selected_model': session['selected_model'],
            'custom_system_prompt': session['custom_system_prompt'],
            'current_role_name': session['current_role_name'],
            'tz_data': session['tz_data'],
            'tz_complete': session['tz_complete'],
            'last_system_prompt': session['last_system_prompt'],
            'compression_enabled': COMPRESSION_CONFIG['enable_compression']
        })

        memory.update_token_stats(session['session_id'], 0, 0, 0.0)
        memory.update_compression_stats(session['session_id'], session['compression_stats'])


def get_model_config(model_key):
    """Получить конфигурацию модели по ключу"""
    return MODELS_CONFIG.get(model_key, MODELS_CONFIG['yandexgpt-lite'])


def estimate_tokens(text):
    """Примерная оценка количества токенов в тексте"""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def analyze_response_behavior(input_tokens, output_tokens, temperature, response_text, model_config):
    """Анализ поведения модели на основе токенов и температуры"""
    try:
        input_tokens = int(input_tokens) if input_tokens is not None else 0
        output_tokens = int(output_tokens) if output_tokens is not None else 0
        temperature = float(temperature) if temperature is not None else 0.7
    except (ValueError, TypeError) as e:
        logger.error(f"Ошибка преобразования типов: {e}")
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

    if input_tokens > 0:
        token_ratio = output_tokens / input_tokens
        behavior_analysis['token_ratio'] = round(token_ratio, 2)

        if input_tokens < 50:
            if output_tokens < 30:
                behavior_analysis['efficiency_score'] = 30
            elif output_tokens > 500:
                behavior_analysis['efficiency_score'] = 90
            else:
                behavior_analysis['efficiency_score'] = 70
        else:
            if token_ratio > 2:
                behavior_analysis['efficiency_score'] = 90
            elif token_ratio > 1:
                behavior_analysis['efficiency_score'] = 80
            elif token_ratio > 0.5:
                behavior_analysis['efficiency_score'] = 60
            else:
                behavior_analysis['efficiency_score'] = 40

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

    input_text = ' '.join([msg['text'] for msg in messages])
    estimated_input_tokens = estimate_tokens(input_text)

    start_time = time.time()
    response = requests.post(YANDEX_API_BASE_URL, headers=headers, json=payload, timeout=60)
    end_time = time.time()
    execution_time = end_time - start_time

    if response.status_code != 200:
        logger.error(f"API Error {response.status_code}: {response.text}")
        return None, execution_time, estimated_input_tokens, 0, 0.0

    result = response.json()

    input_tokens = estimated_input_tokens
    output_tokens = 0

    try:
        response_text = ""
        if 'result' in result and 'alternatives' in result['result']:
            response_text = result['result']['alternatives'][0]['message']['text']
        elif 'alternatives' in result:
            response_text = result['alternatives'][0]['message']['text']
        else:
            response_text = extract_message_from_response(result)

        output_tokens = estimate_tokens(response_text)

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
        output_tokens = estimate_tokens(extract_message_from_response(result))

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

    compression_messages = [
        {'role': 'system', 'text': SYSTEM_PROMPTS['history_compression']},
        {'role': 'user',
         'text': f"Создай краткое изложение этого диалога:\n\n{format_messages_for_compression(messages_to_compress)}"}
    ]

    model_config = get_model_config(model_key)

    result, execution_time, input_tokens, output_tokens, cost = call_yandex_api(
        model_config=model_config,
        messages=compression_messages,
        temperature=0.3,
        max_tokens=COMPRESSION_CONFIG['max_summary_tokens']
    )

    if result is None:
        return "", 0, 0, 0.0

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

    if len(full_history) >= COMPRESSION_CONFIG['compression_interval']:
        return True

    total_tokens = sum(estimate_tokens(msg['text']) for msg in full_history)
    if total_tokens > 2000:
        return True

    return False


def get_compression_candidate(full_history, last_compressed_id):
    """Получить сообщения для компрессии"""
    keep_last = COMPRESSION_CONFIG['keep_last_messages']

    uncompressed_messages = []
    for i, msg in enumerate(full_history):
        if i <= last_compressed_id:
            continue
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

    if not compressed_history:
        return full_history[-keep_last * 2:] if len(full_history) > keep_last * 2 else full_history

    last_messages = full_history[-keep_last:] if len(full_history) > keep_last else full_history

    effective_history = []

    for summary in compressed_history:
        effective_history.append({
            'role': 'system',
            'text': f"📚 Краткое содержание предыдущего диалога: {summary}"
        })

    effective_history.extend(last_messages)

    return effective_history


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
                           compression_stats=session.get('compression_stats', {}),
                           memory_enabled=memory is not None)


@app.route('/send_message', methods=['POST'])
def send_message():
    """Обработка отправки сообщения"""
    try:
        logger.info(f"Получен запрос: {request.json}")
        user_message = request.json.get('message', '').strip()
        require_json = request.json.get('require_json', False)
        model_key = request.json.get('model', 'yandexgpt-lite')

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

        session_id = session['session_id']

        session['temperature'] = str(temperature_val)
        session['selected_model'] = model_key

        if memory:
            memory.save_session(session_id, {
                'chat_mode': session['chat_mode'],
                'temperature': session['temperature'],
                'selected_model': session['selected_model'],
                'custom_system_prompt': session['custom_system_prompt'],
                'current_role_name': session['current_role_name'],
                'tz_data': session['tz_data'],
                'tz_complete': session['tz_complete'],
                'last_system_prompt': session['last_system_prompt'],
                'compression_enabled': COMPRESSION_CONFIG['enable_compression']
            })

        # Обработка команд
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
        elif user_message.lower() == '/memory_stats':
            return show_memory_stats()
        elif user_message.lower() == '/memory_sessions':
            return list_sessions_command()
        elif user_message.lower() == '/memory_export':
            return export_current_session()
        elif user_message.lower() == '/memory_cleanup':
            return cleanup_sessions_command()

        chat_mode = session.get('chat_mode', 'default')
        tz_complete = session.get('tz_complete', False)

        if tz_complete and chat_mode != 'tz_collection':
            session['tz_complete'] = False
            tz_complete = False

        if tz_complete:
            return jsonify({
                'error': 'Сбор ТЗ завершен. Начните новый сбор или очистите историю.',
                'tz_data': session.get('tz_data')
            }), 400

        if require_json and chat_mode not in ['tz_collection']:
            actual_mode = 'json_format'
        else:
            actual_mode = chat_mode

        full_history = session.get('chat_history', [])
        full_history.append({'role': 'user', 'text': user_message})

        compression_occurred = False
        compression_cost = 0.0
        compression_input_tokens = 0
        compression_output_tokens = 0

        if COMPRESSION_CONFIG['enable_compression'] and should_compress_history(full_history):
            candidate_messages = get_compression_candidate(
                full_history,
                session.get('last_compression_message_id', -1)
            )

            if candidate_messages and len(candidate_messages) >= 2:
                summary, comp_input, comp_output, comp_cost = compress_history(
                    candidate_messages,
                    COMPRESSION_CONFIG['summary_model']
                )

                if summary:
                    compressed_history = session.get('compressed_history', [])
                    compressed_history.append(summary)
                    session['compressed_history'] = compressed_history

                    last_msg_index = full_history.index(candidate_messages[-1])
                    session['last_compression_message_id'] = last_msg_index

                    original_tokens = sum(estimate_tokens(msg['text']) for msg in candidate_messages)
                    compressed_tokens = estimate_tokens(summary)
                    update_compression_stats(original_tokens, compressed_tokens)

                    compression_occurred = True
                    compression_cost = comp_cost
                    compression_input_tokens = comp_input
                    compression_output_tokens = comp_output

                    logger.info(
                        f"✅ Сжатие истории: {len(candidate_messages)} сообщений -> summary ({compressed_tokens} токенов)")

        effective_history = get_effective_history()

        if session.get('custom_system_prompt'):
            system_prompt = session['custom_system_prompt']
        else:
            system_prompt = SYSTEM_PROMPTS.get(actual_mode, SYSTEM_PROMPTS['default'])

        session['last_system_prompt'] = system_prompt

        messages = [{'role': 'system', 'text': system_prompt}]

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

        model_config = get_model_config(model_key)

        if actual_mode == 'json_format':
            max_tokens = min(1500, int(model_config['max_tokens']))
        elif actual_mode == 'tz_collection':
            max_tokens = min(2000, int(model_config['max_tokens']))
        elif session.get('custom_system_prompt'):
            max_tokens = min(2000, int(model_config['max_tokens']))
        else:
            max_tokens = min(1500, int(model_config['max_tokens']))

        result, execution_time, input_tokens, output_tokens, cost = call_yandex_api(
            model_config, messages, temperature_val, max_tokens
        )

        if result is None:
            return jsonify({'error': 'Ошибка при вызове API модели'}), 500

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

        try:
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
        except (ValueError, TypeError):
            input_tokens = estimate_tokens(' '.join([msg['text'] for msg in messages]))
            output_tokens = estimate_tokens(assistant_message)

        total_input = input_tokens + compression_input_tokens
        total_output = output_tokens + compression_output_tokens
        total_cost_api = cost + compression_cost

        behavior_analysis = analyze_response_behavior(
            input_tokens, output_tokens, temperature_val, assistant_message, model_config
        )

        session['total_input_tokens'] = session.get('total_input_tokens', 0) + total_input
        session['total_output_tokens'] = session.get('total_output_tokens', 0) + total_output
        session['total_cost'] = session.get('total_cost', 0.0) + total_cost_api

        full_history.append({'role': 'assistant', 'text': assistant_message})

        if len(full_history) > 100:
            full_history = full_history[-100:]

        session['chat_history'] = full_history
        session.modified = True

        if memory:
            memory.save_message(
                session_id=session_id,
                message_index=len(full_history) - 2,
                role='user',
                content=user_message,
                tokens=0,
                cost=0.0,
                model=model_key,
                temperature=temperature_val
            )

            memory.save_message(
                session_id=session_id,
                message_index=len(full_history) - 1,
                role='assistant',
                content=assistant_message,
                tokens=output_tokens,
                cost=cost,
                model=model_key,
                temperature=temperature_val
            )

            memory.save_intermediate_result(
                session_id=session_id,
                result_type='api_response',
                data=result,
                metadata={
                    'model': model_key,
                    'temperature': temperature_val,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'execution_time': execution_time
                }
            )

            memory.update_token_stats(
                session_id=session_id,
                input_tokens=total_input,
                output_tokens=total_output,
                cost=total_cost_api
            )

            if compression_occurred:
                memory.update_compression_stats(
                    session_id=session_id,
                    stats=session['compression_stats']
                )

                memory.save_compressed_history(
                    session_id=session_id,
                    summary=summary,
                    original_messages_count=len(candidate_messages),
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens
                )

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
            },
            'memory_enabled': memory is not None,
            'session_id': session_id
        }

        if compression_occurred:
            response_data['compression_stats'] = session.get('compression_stats', {})

        logger.info(f"Модель: {model_key}, Вход: {input_tokens}, Выход: {output_tokens}, "
                    f"Время: {execution_time:.2f}с, Стоимость: {cost:.5f} руб")

        return jsonify(response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return jsonify({'error': 'Ошибка соединения с YandexGPT API.'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500


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

        if memory:
            memory.save_session(session['session_id'], {
                'chat_mode': session['chat_mode'],
                'custom_system_prompt': session['custom_system_prompt'],
                'current_role_name': session['current_role_name']
            })

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

        if memory:
            memory.save_session(session['session_id'], {
                'chat_mode': session['chat_mode'],
                'custom_system_prompt': session['custom_system_prompt'],
                'current_role_name': session['current_role_name']
            })

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

    if memory:
        memory.save_session(session['session_id'], {
            'chat_mode': session['chat_mode'],
            'custom_system_prompt': session['custom_system_prompt'],
            'current_role_name': session['current_role_name']
        })

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
"""

    return jsonify({
        'response': stats_text,
        'is_command': True,
        'current_role': session.get('current_role_name', 'Обычный ассистент'),
        'history_count': len(session.get('chat_history', []))
    })


def show_memory_stats():
    """Показать статистику внешней памяти"""
    if not memory:
        return jsonify({
            'response': "❌ Внешняя память не инициализирована",
            'is_command': True
        })

    try:
        stats = memory.get_database_stats()

        stats_text = f"""
💾 **Статистика внешней памяти (SQLite):**

• Всего сессий: {stats.get('total_sessions', 0)}
• Всего сообщений: {stats.get('total_messages', 0)}
• Сжатых историй: {stats.get('total_compressed_histories', 0)}
• Всего токенов: {stats.get('total_tokens', 0)}
• Размер БД: {stats.get('database_size_mb', 0)} МБ

🔗 **Команды управления памятью:**
• `/memory_sessions` - список сессий
• `/memory_export` - экспорт текущей сессии
• `/memory_cleanup` - очистка старых сессий
"""

        return jsonify({
            'response': stats_text,
            'is_command': True,
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(session.get('chat_history', []))
        })
    except Exception as e:
        logger.error(f"Error getting memory stats: {str(e)}")
        return jsonify({
            'response': f"❌ Ошибка получения статистики памяти: {str(e)}",
            'is_command': True
        })


def list_sessions_command():
    """Команда для вывода списка сессий"""
    if not memory:
        return jsonify({
            'response': "❌ Внешняя память не инициализирована",
            'is_command': True
        })

    try:
        sessions = memory.get_session_list(5)

        if not sessions:
            return jsonify({
                'response': "📂 Нет сохраненных сессий во внешней памяти",
                'is_command': True
            })

        sessions_text = "📂 **Последние сессии во внешней памяти:**\n\n"
        for i, sess in enumerate(sessions, 1):
            sessions_text += f"{i}. **{sess['session_id'][:8]}...** - {sess['message_count']} сообщений, {sess['total_tokens']} токенов\n"
            sessions_text += f"   Режим: {sess['chat_mode']}, Роль: {sess['current_role_name']}\n"
            sessions_text += f"   Обновлено: {sess['updated_at'][:16]}\n\n"

        return jsonify({
            'response': sessions_text,
            'is_command': True,
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(session.get('chat_history', []))
        })
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({
            'response': f"❌ Ошибка получения списка сессий: {str(e)}",
            'is_command': True
        })


def export_current_session():
    """Команда для экспорта текущей сессии"""
    if not memory:
        return jsonify({
            'response': "❌ Внешняя память не инициализирована",
            'is_command': True
        })

    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({
                'response': "❌ Нет активной сессии",
                'is_command': True
            })

        export_data = memory.export_session(session_id)
        if not export_data:
            return jsonify({
                'response': "❌ Ошибка экспорта сессии",
                'is_command': True
            })

        # Сохраняем экспорт в файл
        filename = f"session_export_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(export_data)

        return jsonify({
            'response': f"✅ Сессия экспортирована в файл: `{filename}`",
            'is_command': True,
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(session.get('chat_history', []))
        })
    except Exception as e:
        logger.error(f"Error exporting session: {str(e)}")
        return jsonify({
            'response': f"❌ Ошибка экспорта сессии: {str(e)}",
            'is_command': True
        })


def cleanup_sessions_command():
    """Команда для очистки старых сессий"""
    if not memory:
        return jsonify({
            'response': "❌ Внешняя память не инициализирована",
            'is_command': True
        })

    try:
        deleted_count = memory.cleanup_old_sessions(7)

        return jsonify({
            'response': f"✅ Удалено {deleted_count} сессий старше 7 дней",
            'is_command': True,
            'current_role': session.get('current_role_name', 'Обычный ассистент'),
            'history_count': len(session.get('chat_history', []))
        })
    except Exception as e:
        logger.error(f"Error cleaning up sessions: {str(e)}")
        return jsonify({
            'response': f"❌ Ошибка очистки сессий: {str(e)}",
            'is_command': True
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
/memory_stats - показать статистику памяти
/memory_sessions - список сохраненных сессий
/memory_export - экспорт текущей сессии
/memory_cleanup - очистка старых сессий
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
            session['compressed_history'] = []
            session['last_compression_message_id'] = -1
        else:
            session['custom_system_prompt'] = None
            session['current_role_name'] = 'Обычный ассистент'

        session['chat_mode'] = mode
        session.modified = True

        if memory:
            memory.save_session(session['session_id'], {
                'chat_mode': session['chat_mode'],
                'custom_system_prompt': session['custom_system_prompt'],
                'current_role_name': session['current_role_name'],
                'tz_data': session['tz_data'],
                'tz_complete': session['tz_complete']
            })

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

    session_id = session['session_id']

    if memory:
        memory.delete_session(session_id)
        memory.save_session(session_id, {
            'chat_mode': session.get('chat_mode', 'default'),
            'temperature': session.get('temperature', 0.7),
            'selected_model': session.get('selected_model', 'yandexgpt-lite'),
            'custom_system_prompt': session.get('custom_system_prompt'),
            'current_role_name': session.get('current_role_name', 'Обычный ассистент'),
            'compression_enabled': COMPRESSION_CONFIG['enable_compression']
        })
        memory.update_token_stats(session_id, 0, 0, 0.0)
        memory.update_compression_stats(session_id, {
            'total_compressions': 0,
            'tokens_saved': 0,
            'original_tokens': 0,
            'compressed_tokens': 0,
            'compression_ratio': 0.0
        })

    session['chat_history'] = []
    session['compressed_history'] = []
    session['tz_data'] = None
    session['tz_complete'] = False
    session['total_input_tokens'] = 0
    session['total_output_tokens'] = 0
    session['total_cost'] = 0.0
    session['compression_stats'] = {
        'total_compressions': 0,
        'tokens_saved': 0,
        'original_tokens': 0,
        'compressed_tokens': 0,
        'compression_ratio': 0.0
    }
    session['last_compression_message_id'] = -1
    session.modified = True

    system_message = "🗑️ История диалога, компрессия и статистика очищены. Внешняя память также очищена."

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

    enable = request.json.get('enable')
    if enable is None:
        COMPRESSION_CONFIG['enable_compression'] = not COMPRESSION_CONFIG['enable_compression']
    else:
        COMPRESSION_CONFIG['enable_compression'] = bool(enable)

    status = "включена" if COMPRESSION_CONFIG['enable_compression'] else "выключена"

    if memory:
        memory.save_session(session['session_id'], {
            'compression_enabled': COMPRESSION_CONFIG['enable_compression']
        })

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
        'compression_enabled': COMPRESSION_CONFIG['enable_compression'],
        'memory_enabled': memory is not None
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
        'compression_enabled': COMPRESSION_CONFIG['enable_compression'],
        'memory_enabled': memory is not None
    })


@app.route('/get_models', methods=['GET'])
def get_models():
    """Получить список доступных моделей"""
    return jsonify(MODELS_CONFIG)


# Новые эндпоинты для управления памятью

@app.route('/memory/stats', methods=['GET'])
def get_memory_stats_api():
    """Получить статистику внешней памяти"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        stats = memory.get_database_stats()
        return jsonify({
            'success': True,
            'stats': stats,
            'memory_enabled': True
        })
    except Exception as e:
        logger.error(f"Ошибка получения статистики памяти: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory/sessions', methods=['GET'])
def get_sessions_list_api():
    """Получить список всех сохраненных сессий"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        limit = request.args.get('limit', 100, type=int)
        sessions = memory.get_session_list(limit)
        return jsonify({
            'success': True,
            'sessions': sessions,
            'count': len(sessions)
        })
    except Exception as e:
        logger.error(f"Ошибка получения списка сессий: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory/session/<session_id>', methods=['GET'])
def load_session_by_id(session_id):
    """Загрузить конкретную сессию по ID"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        session_data = memory.load_session(session_id)
        if not session_data:
            return jsonify({'error': 'Сессия не найдена'}), 404

        return jsonify({
            'success': True,
            'session': session_data
        })
    except Exception as e:
        logger.error(f"Ошибка загрузки сессии: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory/session/<session_id>/export', methods=['GET'])
def export_session_api(session_id):
    """Экспортировать сессию в JSON"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        format_type = request.args.get('format', 'json')
        export_data = memory.export_session(session_id, format_type)

        if not export_data:
            return jsonify({'error': 'Сессия не найдена или ошибка экспорта'}), 404

        if format_type == 'json':
            response = Response(
                export_data,
                status=200,
                mimetype='application/json'
            )
            response.headers['Content-Disposition'] = f'attachment; filename=session_{session_id}.json'
            return response
        else:
            return jsonify({'error': 'Неподдерживаемый формат экспорта'}), 400
    except Exception as e:
        logger.error(f"Ошибка экспорта сессии: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory/session/<session_id>', methods=['DELETE'])
def delete_session_api(session_id):
    """Удалить сессию из внешней памяти"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        memory.delete_session(session_id)
        return jsonify({
            'success': True,
            'message': f'Сессия {session_id} удалена'
        })
    except Exception as e:
        logger.error(f"Ошибка удаления сессии: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory/cleanup', methods=['POST'])
def cleanup_old_sessions_api():
    """Очистить старые сессии"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        days_old = request.json.get('days_old', 30)
        deleted_count = memory.cleanup_old_sessions(days_old)

        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Удалено {deleted_count} сессий старше {days_old} дней'
        })
    except Exception as e:
        logger.error(f"Ошибка очистки старых сессий: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/memory/last_session', methods=['GET'])
def get_last_session_info():
    """Получить информацию о последней сессии"""
    if not memory:
        return jsonify({'error': 'Внешняя память не инициализирована'}), 500

    try:
        last_session_id = memory.get_last_session_id()
        if not last_session_id:
            return jsonify({
                'success': False,
                'message': 'Нет сохраненных сессий'
            })

        session_data = memory.load_session(last_session_id)
        if not session_data:
            return jsonify({
                'success': False,
                'message': 'Не удалось загрузить сессию'
            })

        return jsonify({
            'success': True,
            'session_id': last_session_id,
            'message_count': len(session_data.get('chat_history', [])),
            'updated_at': datetime.now().isoformat(),
            'chat_mode': session_data.get('chat_mode', 'default'),
            'current_role': session_data.get('current_role_name', 'Обычный ассистент')
        })
    except Exception as e:
        logger.error(f"Ошибка получения последней сессии: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Сервис для обслуживания статических файлов"""
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Создаем папки если их нет
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    app.run(debug=True, host='0.0.0.0', port=5000)