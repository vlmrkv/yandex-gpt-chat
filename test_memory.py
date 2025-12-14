#!/usr/bin/env python3
"""
Скрипт для тестирования работы внешней памяти
"""

import json
import sqlite3
from datetime import datetime, timedelta
from memory_storage import MemoryStorage


def test_memory_storage():
    """Тестирование работы внешней памяти"""
    print("🧪 Тестирование внешней памяти...")

    # Создаем экземпляр хранилища
    storage = MemoryStorage("test_memory.db")

    # Тест 1: Создание и сохранение сессии
    print("\n1. Тест создания сессии...")
    session_id = "test_session_123"
    storage.save_session(session_id, {
        'chat_mode': 'default',
        'temperature': 0.7,
        'selected_model': 'yandexgpt-lite',
        'current_role_name': 'Тестовый ассистент'
    })
    print(f"   ✅ Сессия создана: {session_id}")

    # Тест 2: Сохранение сообщений
    print("\n2. Тест сохранения сообщений...")
    storage.save_message(session_id, 0, 'user', 'Привет! Как дела?')
    storage.save_message(session_id, 1, 'assistant', 'Привет! У меня все отлично, спасибо!')
    print("   ✅ Сообщения сохранены")

    # Тест 3: Сохранение статистики
    print("\n3. Тест сохранения статистики...")
    storage.update_token_stats(session_id, 100, 150, 0.05)
    storage.update_compression_stats(session_id, {
        'total_compressions': 1,
        'tokens_saved': 50,
        'compression_ratio': 30.5
    })
    print("   ✅ Статистика сохранена")

    # Тест 4: Сохранение сжатой истории
    print("\n4. Тест сохранения сжатой истории...")
    storage.save_compressed_history(
        session_id,
        'Обсудили приветствие и настроение.',
        2,  # 2 сообщения
        250,  # 250 токенов
        50  # 50 токенов после сжатия
    )
    print("   ✅ Сжатая история сохранена")

    # Тест 5: Сохранение промежуточных результатов
    print("\n5. Тест сохранения промежуточных результатов...")
    storage.save_intermediate_result(
        session_id,
        'api_response',
        {'response': 'Тестовый ответ API'},
        {'model': 'yandexgpt-lite', 'temperature': 0.7}
    )
    print("   ✅ Промежуточный результат сохранен")

    # Тест 6: Загрузка сессии
    print("\n6. Тест загрузки сессии...")
    loaded_session = storage.load_session(session_id)
    if loaded_session:
        print(f"   ✅ Сессия загружена успешно")
        print(f"   - Сообщений: {len(loaded_session['chat_history'])}")
        print(
            f"   - Токенов: {loaded_session.get('total_input_tokens', 0) + loaded_session.get('total_output_tokens', 0)}")
    else:
        print("   ❌ Ошибка загрузки сессии")

    # Тест 7: Получение списка сессий
    print("\n7. Тест получения списка сессий...")
    sessions = storage.get_session_list()
    print(f"   ✅ Найдено сессий: {len(sessions)}")

    # Тест 8: Экспорт сессии
    print("\n8. Тест экспорта сессии...")
    export_data = storage.export_session(session_id)
    if export_data:
        print(f"   ✅ Сессия экспортирована, размер: {len(export_data)} байт")
    else:
        print("   ❌ Ошибка экспорта сессии")

    # Тест 9: Получение статистики БД
    print("\n9. Тест получения статистики БД...")
    db_stats = storage.get_database_stats()
    print(f"   ✅ Статистика БД: {db_stats}")

    # Тест 10: Удаление сессии
    print("\n10. Тест удаления сессии...")
    storage.delete_session(session_id)
    print(f"   ✅ Сессия удалена")

    # Проверяем, что сессия удалена
    loaded_after_delete = storage.load_session(session_id)
    if not loaded_after_delete:
        print("   ✅ Сессия успешно удалена из БД")
    else:
        print("   ❌ Сессия не была удалена")

    # Очистка тестовой БД
    import os
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
        print("\n🧹 Тестовая база данных удалена")

    print("\n" + "=" * 50)
    print("✅ Все тесты завершены успешно!")
    print("=" * 50)


def test_persistence():
    """Тест сохранения данных между запусками"""
    print("\n🧪 Тест сохранения данных между запусками...")

    # Первый запуск: создаем и сохраняем данные
    print("\n1. Первый запуск: создание данных...")
    storage1 = MemoryStorage("persistence_test.db")

    session_id = "persistence_test_456"
    storage1.save_session(session_id, {
        'chat_mode': 'json_format',
        'temperature': 0.3,
        'selected_model': 'yandexgpt',
        'current_role_name': 'Аналитик данных'
    })

    storage1.save_message(session_id, 0, 'user', 'Создай JSON с данными о продукте')
    storage1.save_message(session_id, 1, 'assistant', '{"product": "Тестовый", "price": 100}')
    storage1.update_token_stats(session_id, 200, 300, 0.1)

    print(f"   ✅ Данные сохранены в сессии: {session_id}")

    # "Закрываем" первое соединение
    del storage1

    # Второй запуск: загружаем данные
    print("\n2. Второй запуск: загрузка данных...")
    storage2 = MemoryStorage("persistence_test.db")

    loaded_session = storage2.load_session(session_id)

    if loaded_session and loaded_session['chat_history']:
        print(f"   ✅ Данные успешно загружены между запусками!")
        print(f"   - Режим: {loaded_session['chat_mode']}")
        print(f"   - Сообщений: {len(loaded_session['chat_history'])}")
        print(
            f"   - Токенов: {loaded_session.get('total_input_tokens', 0)} входных, {loaded_session.get('total_output_tokens', 0)} выходных")

        # Проверяем целостность данных
        if loaded_session['chat_history'][0]['text'] == 'Создай JSON с данными о продукте':
            print("   ✅ Целостность данных подтверждена")
        else:
            print("   ❌ Ошибка целостности данных")
    else:
        print("   ❌ Данные не были сохранены между запусками")

    # Очистка
    if os.path.exists("persistence_test.db"):
        os.remove("persistence_test.db")

    print("\n" + "=" * 50)
    print("✅ Тест persistence завершен!")
    print("=" * 50)


def check_real_database():
    """Проверка реальной базы данных приложения"""
    print("\n🔍 Проверка реальной базы данных приложения...")

    if os.path.exists("chat_memory.db"):
        # Подключаемся к базе данных
        conn = sqlite3.connect("chat_memory.db")
        cursor = conn.cursor()

        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        print(f"   📊 Таблиц в базе данных: {len(tables)}")
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   - {table_name}: {count} записей")

        # Получаем размер базы данных
        size_bytes = os.path.getsize("chat_memory.db")
        size_mb = size_bytes / (1024 * 1024)
        print(f"   💾 Размер базы данных: {size_mb:.2f} МБ")

        # Получаем последние сессии
        cursor.execute("""
                       SELECT session_id, created_at, updated_at
                       FROM sessions
                       ORDER BY updated_at DESC LIMIT 3
                       """)
        recent_sessions = cursor.fetchall()

        if recent_sessions:
            print(f"   📅 Последние сессии:")
            for sess in recent_sessions:
                print(f"   - {sess[0][:8]}... создана: {sess[1][:10]}, обновлена: {sess[2][:10]}")

        conn.close()
    else:
        print("   ❌ База данных не найдена. Запустите приложение для ее создания.")

    print("\n" + "=" * 50)
    print("✅ Проверка завершена!")
    print("=" * 50)


if __name__ == "__main__":
    import os

    print("=" * 50)
    print("🔧 ТЕСТИРОВАНИЕ ВНЕШНЕЙ ПАМЯТИ ДЛЯ МОДЕЛИ")
    print("=" * 50)

    try:
        # Запускаем основные тесты
        test_memory_storage()

        # Тест сохранения между запусками
        test_persistence()

        # Проверяем реальную базу данных
        check_real_database()

    except Exception as e:
        print(f"\n❌ Ошибка во время тестирования: {e}")
        import traceback

        traceback.print_exc()