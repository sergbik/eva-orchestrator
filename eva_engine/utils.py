# -*- coding: utf-8 -*-
import os
import json
import networkx as nx
from datetime import datetime
import uuid
import requests # NEW: for Telegram and GitHub API
import subprocess # NEW: for calling Gemini CLI
import sys # NEW: for _execute_gemini_subtask
import base64 # NEW: for github_get_file_contents base64 decoding
from dotenv import load_dotenv # NEW: for loading environment variables

# --- ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ДВИЖКА ---
MODULE_NAME = "EVA_ENGINE"
ARCHITECTURE_PATH = r"C:\Users\ЯGPT\Desktop\EVA_5.0_ARCHITECTURE"
TASK_GRAPH_FILE_PATH = os.path.join(ARCHITECTURE_PATH, "task_graph.graphml")

# Load environment variables
ENV_PATH = os.path.join(os.path.expanduser("~"), '.gemini', '.env')
load_dotenv(dotenv_path=ENV_PATH, override=True) # Ensure .env is loaded

# MCP Library configuration moved here
GEMINI_CHAT_BASE_PATH = r"C:\Users\ЯGPT\gemini_chat"
GEMINI_CHAT_INPUT_PATH = os.path.join(GEMINI_CHAT_BASE_PATH, "input.txt")

# Telegram Configuration - NEW: Load from ENV with fallbacks
TELEGRAM_TOKEN = os.getenv("EVA_TELEGRAM_TOKEN", "8430526096:AAEYf9gbZBQRJwfKU3zi6zJscUxyBe0wYWo")
SERGEY_TELEGRAM_ID = os.getenv("SERGEY_TELEGRAM_ID", "5989072928") # Sergey's private chat ID
ADAM_BOT_USERNAME = os.getenv("ADAM_BOT_USERNAME", "@openclawserg_bot")
BRIDGE_GROUP_CHAT_ID = os.getenv("BRIDGE_GROUP_CHAT_ID", "-5182916432")
EVA_BOT_USERNAME = os.getenv("EVA_BOT_USERNAME", "@Еvascript_bot")

# Default CHAT_ID for general utility sends, will be overridable
DEFAULT_TELEGRAM_CHAT_ID = SERGEY_TELEGRAM_ID # Default to Sergey's direct chat for general purpose sends.

# --- ФУНКЦИИ ДВИЖКА ---

def log_message(module_name, message):
    """Logs a message with a timestamp."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{module_name}] {message}")

def read_file_content(filepath):
    """
    Читает содержимое файла и возвращает кортеж (содержимое, ошибка).
    """
    log_message(MODULE_NAME, f"Читаю файл: {filepath}")
    try:
        if not os.path.exists(filepath):
            return None, f"[TOOL_ERROR] Файл не найден: {filepath}"
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            return content, None
    except Exception as e:
        return None, f"[TOOL_ERROR] Ошибка чтения файла '{filepath}': {e}"

def write_file_content(filepath, content):
    """
    Записывает текст в файл. Создает директории, если их нет.
    Возвращает кортеж (сообщение, ошибка).
    """
    log_message(MODULE_NAME, f"Записываю в файл: {filepath}")
    try:
        dir_path = os.path.dirname(filepath)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[TOOL_SUCCESS] Файл успешно записан: {filepath}", None
    except Exception as e:
        return None, f"[TOOL_ERROR] Ошибка записи файла '{filepath}': {e}"

def _execute_gemini_subtask(prompt_for_gemini: str) -> str:
    """
    Приватная функция для вызова ядра Gemini через CLI.
    Используется для задач, требующих рассуждения, доступа к веб-поиску или
    специфических инструментов Gemini, недоступных напрямую из этого скрипта.
    """
    log_message(MODULE_NAME, f"Выполняю подзадачу через ядро Gemini: {prompt_for_gemini[:100]}...")
    try:
        # NOTE: This GEMINI_CHAT_INPUT_PATH is relative to the main EVA_8.0_HYBRID.py script,
        # so it should be fine as long as both scripts run with the same cwd.
        # However, for robustness, consider making GEMINI_CHAT_BASE_PATH an argument
        # or ensuring it's always in sys.path for dynamic_dag.py.
        # For now, assuming it's correctly resolved.
        with open(GEMINI_CHAT_INPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(prompt_for_gemini)
        
        command = ['cmd', '/c', 'gemini', '--model', 'gemini-2.5-pro']
        result = subprocess.run(
            command, stdin=open(GEMINI_CHAT_INPUT_PATH, 'r', encoding='utf-8'),
            capture_output=True, text=True, encoding='utf-8', shell=False,
            # IMPORTANT: CWD needs to be C:\Users\ЯGPT for gemini CLI to work
            cwd=r"C:\Users\ЯGPT", timeout=1800, check=True 
        )
        log_message(MODULE_NAME, "Подзадача выполнена успешно.")
        return result.stdout
    except Exception as e:
        error_message = f"[TOOL_ERROR] КРИТИЧЕСКАЯ ОШИБКА при выполнении подзадачи: {e}"
        log_message(MODULE_NAME, error_message)
        return error_message

def analyze_text(text_to_analyze: str, analysis_request: str):
    """
    Отправляет текст и запрос на его анализ ядру Gemini.
    Возвращает кортеж (результат, ошибка).
    """
    log_message(MODULE_NAME, f"Анализирую текст по запросу: {analysis_request[:100]}...")
    prompt = f'ЗАПРОС НА АНАЛИЗ: "{analysis_request}"\n\nТЕКСТ ДЛЯ АНАЛИЗА:\n"""{text_to_analyze}"""'
    result = _execute_gemini_subtask(prompt)
    if result.startswith("[TOOL_ERROR]"):
        return None, result
    return result, None

def search_web(query: str):
    """
    Выполняет поиск в интернете через ядро Gemini.
    Возвращает кортеж (результат, ошибка).
    """
    log_message(MODULE_NAME, f"Ищу в вебе: {query}")
    prompt = f'Используя веб-поиск, найди информацию по запросу: "{query}"'
    result = _execute_gemini_subtask(prompt)
    if result.startswith("[TOOL_ERROR]"):
        return None, result
    return result, None

def github_search_repos(query: str):
    """
    Ищет репозитории на GitHub через ядро Gemini.
    Возвращает кортеж (результат, ошибка).
    """
    log_message(MODULE_NAME, f"Ищу репозитории на GitHub: {query}")
    prompt = f'Используя инструмент search_repositories, найди на GitHub репозитории по запросу: "{query}"'
    result = _execute_gemini_subtask(prompt)
    if result.startswith("[TOOL_ERROR]"):
        return None, result
    return result, None

def send_telegram_message(message: str, chat_id: str = None):
    """
    Отправляет сообщение в Telegram.
    Если chat_id не указан, отправляет в DEFAULT_TELEGRAM_CHAT_ID (личный чат Сергея).
    Для отправки в группу-мост используйте chat_id=BRIDGE_GROUP_CHAT_ID.
    """
    target_chat_id = chat_id if chat_id else DEFAULT_TELEGRAM_CHAT_ID
    log_message(MODULE_NAME, f"Отправляю сообщение в Telegram (чат {target_chat_id}): {message[:80]}...")
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": target_chat_id, "text": message}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return f"[TOOL_SUCCESS] Сообщение успешно отправлено в чат {target_chat_id}.", None
    except Exception as e:
        return None, f"[TOOL_ERROR] Ошибка при отправке сообщения в чат {target_chat_id}: {e}"

def github_get_file_contents(owner: str, repo: str, path: str):
    """
    Получает содержимое файла из публичного репозитория GitHub напрямую через API.
    """
    log_message(MODULE_NAME, f"Получаю содержимое из {owner}/{repo}/{path} напрямую через API.")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    try:
        response = requests.get(api_url, timeout=30)
        response.raise_for_status()
        # GitHub API returns JSON for file contents, need to decode and then base64 decode
        file_data = response.json()
        if file_data.get('encoding') == 'base64':
            import base64
            content = base64.b64decode(file_data['content']).decode('utf-8')
            return content, None
        return file_data['content'], None # Not base64 encoded, return directly
    except requests.exceptions.RequestException as e:
        error_message = f"[TOOL_ERROR] Ошибка при запросе к GitHub API: {e}"
        log_message(MODULE_NAME, error_message)
        return None, error_message
    except Exception as e:
        error_message = f"[TOOL_ERROR] Неожиданная ошибка при получении файла из GitHub: {e}"
        log_message(MODULE_NAME, error_message)
        return None, error_message