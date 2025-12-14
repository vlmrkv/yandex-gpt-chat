import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


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

                # Таблица для промежуточных результатов (сырые данные API, промпты и т.д.)
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

                # Создаем индексы для быстрого поиска
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

                # Проверяем, существует ли сессия
                cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                exists = cursor.fetchone()

                metadata = json.dumps({
                    'tz_data': session_data.get('tz_data'),
                    'tz_complete': session_data.get('tz_complete', False),
                    'last_system_prompt': session_data.get('last_system_prompt'),
                    'compression_enabled': session_data.get('compression_enabled', True)
                })

                if exists:
                    # Обновляем существующую сессию
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
                    # Создаем новую сессию
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
                logger.debug(f"Сессия сохранена: {session_id}")

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
                logger.debug(f"Сообщение сохранено: session={session_id}, index={message_index}")

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
                logger.debug(f"Сжатая история сохранена: session={session_id}")

        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения сжатой истории: {e}")

    def update_token_stats(self, session_id: str, input_tokens: int = 0,
                           output_tokens: int = 0, cost: float = 0.0):
        """Обновить статистику токенов"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Проверяем, существует ли запись
                cursor.execute("SELECT session_id FROM token_stats WHERE session_id = ?", (session_id,))
                exists = cursor.fetchone()

                if exists:
                    # Обновляем существующую запись
                    cursor.execute('''
                                   UPDATE token_stats
                                   SET total_input_tokens  = total_input_tokens + ?,
                                       total_output_tokens = total_output_tokens + ?,
                                       total_cost          = total_cost + ?,
                                       updated_at          = CURRENT_TIMESTAMP
                                   WHERE session_id = ?
                                   ''', (input_tokens, output_tokens, cost, session_id))
                else:
                    # Создаем новую запись
                    cursor.execute('''
                                   INSERT INTO token_stats
                                       (session_id, total_input_tokens, total_output_tokens, total_cost)
                                   VALUES (?, ?, ?, ?)
                                   ''', (session_id, input_tokens, output_tokens, cost))

                conn.commit()
                logger.debug(f"Статистика токенов обновлена: session={session_id}")

        except sqlite3.Error as e:
            logger.error(f"Ошибка обновления статистики токенов: {e}")

    def update_compression_stats(self, session_id: str, stats: Dict[str, Any]):
        """Обновить статистику компрессии"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Проверяем, существует ли запись
                cursor.execute("SELECT session_id FROM compression_stats WHERE session_id = ?", (session_id,))
                exists = cursor.fetchone()

                if exists:
                    # Обновляем существующую запись
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
                    # Создаем новую запись
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
                logger.debug(f"Статистика компрессии обновлена: session={session_id}")

        except sqlite3.Error as e:
            logger.error(f"Ошибка обновления статистики компрессии: {e}")

    def save_intermediate_result(self, session_id: str, result_type: str,
                                 data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Сохранить промежуточный результат (сырой ответ API, промпты и т.д.)"""
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
                logger.debug(f"Промежуточный результат сохранен: session={session_id}, type={result_type}")

        except sqlite3.Error as e:
            logger.error(f"Ошибка сохранения промежуточного результата: {e}")

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Загрузить все данные сессии из базы данных"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Загружаем основную информацию о сессии
                cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
                session_row = cursor.fetchone()

                if not session_row:
                    logger.warning(f"Сессия не найдена: {session_id}")
                    return None

                # Преобразуем строку в словарь
                session_data = dict(session_row)

                # Парсим metadata JSON
                if session_data.get('metadata'):
                    try:
                        metadata = json.loads(session_data['metadata'])
                        session_data.update(metadata)
                    except json.JSONDecodeError as e:
                        logger.error(f"Ошибка парсинга metadata: {e}")

                # Загружаем сообщения
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

                # Загружаем сжатую историю
                cursor.execute('''
                               SELECT summary
                               FROM compressed_history
                               WHERE session_id = ?
                               ORDER BY created_at
                               ''', (session_id,))

                compressed_history = [row['summary'] for row in cursor.fetchall()]

                # Загружаем статистику токенов
                cursor.execute("SELECT * FROM token_stats WHERE session_id = ?", (session_id,))
                token_stats_row = cursor.fetchone()

                # Загружаем статистику компрессии
                cursor.execute("SELECT * FROM compression_stats WHERE session_id = ?", (session_id,))
                compression_stats_row = cursor.fetchone()

                # Собираем результат
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

                # Добавляем статистику
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

                # Удаляем все связанные данные (каскадное удаление)
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

    def get_last_session_id(self):
        """Получить ID последней сессии по времени обновления"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT session_id FROM sessions 
                    ORDER BY updated_at DESC 
                    LIMIT 1
                ''')
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Ошибка получения последней сессии: {e}")
            return None

    def export_session(self, session_id: str, format: str = 'json') -> Optional[str]:
        """Экспортировать сессию в JSON"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return None

            # Добавляем метаданные экспорта
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
                logger.error(f"Неподдерживаемый формат экспорта: {format}")
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

                # Количество сессий
                cursor.execute("SELECT COUNT(*) FROM sessions")
                stats['total_sessions'] = cursor.fetchone()[0]

                # Количество сообщений
                cursor.execute("SELECT COUNT(*) FROM messages")
                stats['total_messages'] = cursor.fetchone()[0]

                # Количество сжатых историй
                cursor.execute("SELECT COUNT(*) FROM compressed_history")
                stats['total_compressed_histories'] = cursor.fetchone()[0]

                # Общий объем токенов
                cursor.execute("SELECT SUM(total_input_tokens + total_output_tokens) FROM token_stats")
                total_tokens = cursor.fetchone()[0]
                stats['total_tokens'] = total_tokens if total_tokens else 0

                # Размер базы данных
                if os.path.exists(self.db_path):
                    stats['database_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)

                return stats

        except sqlite3.Error as e:
            logger.error(f"Ошибка получения статистики базы данных: {e}")
            return {}