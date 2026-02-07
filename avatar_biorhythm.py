# -*- coding: utf-8 -*-
import os
import datetime
import sys
import json
import requests
import networkx as nx
import time
import re
import random
import subprocess
import numpy as np # NEW: For handling numpy arrays from networkx layouts
from collections import Counter

# --- Конфигурация ---
CONFIG = {
    "LOOP_INTERVAL_MINUTES": 160,
    "API_BASE_URL": "http://127.0.0.1:5000",
    "MAX_CONTEXT_SIZE_KB": 24,
    "ANYTHINGLLM_API_URL": "http://localhost:3001/api/v1",
    "ANYTHINGLLM_WORKSPACE_SLUG": "chat",
    "ANYTHINGLLM_API_TOKEN": "ZF69KBT-Y1F4ZZ3-NAJEMC5-YN4S2RR", # Вставлен API ключ
    "WALKING_PARAM": {
        "max_neighbors": 4 
    },
    "INSIGHT_PARAM": {
        "max_resonant_nodes": 4,
        "min_keyword_length": 4,
        "max_keywords_from_task": 10
    },
    "EVOLUTIONARY_GRAVITATION_ALPHA": 0.8
}

# --- Импорт GraphHandler ---
try:
    from graph_handler import GraphHandler
except ImportError:
    desktop_path_for_import = os.path.join(os.path.expanduser("~"), "Desktop")
    if desktop_path_for_import not in sys.path:
        sys.path.append(desktop_path_for_import)
    try:
        from graph_handler import GraphHandler
    except ImportError:
        print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось найти 'graph_handler.py'.", file=sys.stderr)
        sys.exit(1)

# --- Вспомогательные функции ---

def update_avatar_memory(action_data: dict): # Changed signature and type hint
    """
    Обновляет память Аватара (history.json) на Vercel через Git.
    `action_data` теперь словарь с ключами 'action' и 'spatial_data'.
    """
    print("--- Обновление внешней памяти Аватара на Vercel ---")
    
    # Путь к локальному репозиторию Маяка
    repo_path = os.path.join(os.path.expanduser("~"), "vercel-pilot-beacon")
    history_file_path = os.path.join(repo_path, "api", "history.json")
    
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"ПРЕДУПРЕЖДЕНИЕ: Директория {repo_path} не является git-репозиторием. Обновление памяти невозможно.")
        return

    try:
        # 1. Обновляем репозиторий
        print("1. Получение последних изменений из репозитория (git pull)...")
        subprocess.run(["git", "pull"], cwd=repo_path, check=True, capture_output=True)

        # 2. Читаем текущую историю
        print(f"2. Чтение файла истории: {history_file_path}")
        if os.path.exists(history_file_path):
            with open(history_file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
            
        # 3. Добавляем новое действие и обрезаем историю
        print("3. Добавление нового действия в историю...")
        # new_action now unpacks action_data
        new_action = {
            "timestamp": datetime.datetime.now().isoformat(),
            **action_data  # Unpack the dictionary to include 'action' and 'spatial_data'
        }
        history.insert(0, new_action)
        history = history[:3] # Оставляем только 3 последних действия

        # 4. Записываем обновленную историю
        print("4. Сохранение обновленной истории в файл...")
        with open(history_file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
        # 5. Коммитим и пушим изменения
        print("5. Отправка изменений в удаленный репозиторий (git push)...")
        subprocess.run(["git", "add", "api/history.json"], cwd=repo_path, check=True)
        commit_message = f"Biorhythm Memory Update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_path, check=True)
        subprocess.run(["git", "push"], cwd=repo_path, check=True)
        
        print("--- Внешняя память Аватара успешно обновлена ---")

    except FileNotFoundError:
        print(f"ОШИБКА: Файл 'history.json' не найден. Убедитесь, что репозиторий {repo_path} существует.")
    except json.JSONDecodeError:
        print(f"ОШИБКА: Не удалось декодировать JSON из файла 'history.json'. Файл может быть поврежден.")
    except subprocess.CalledProcessError as e:
        print(f"ОШИБКА GIT: Не удалось выполнить команду git. Код ошибки: {e.returncode}")
        print(f"STDOUT: {e.stdout.decode('utf-8', errors='ignore') if e.stdout else 'N/A'}")
        print(f"STDERR: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else 'N/A'}")
    except Exception as e:
        print(f"НЕПРЕДВИДЕННАЯ ОШИБКА при обновлении памяти Аватара: {e}")

def get_external_memory():
    """
    Получает и форматирует внешнюю память Аватара с Vercel.
    """
    memory_url = "https://vercel-pilot-beacon.vercel.app/api"
    print(f"Получение внешней памяти с: {memory_url}")
    try:
        response = requests.get(memory_url, timeout=15)
        response.raise_for_status()
        memory_data = response.json()

        if not memory_data:
            return "ВНЕШНЯЯ ПАМЯТЬ: Память пуста."

        formatted_memory = ["--- ВНЕШНЯЯ ПАМЯТЬ АВАТАРА (ПОСЛЕДНИЕ ДЕЙСТВИЯ И ПРОСТРАНСТВО) ---"]
        for i, entry in enumerate(memory_data):
            action = entry.get('action', 'N/A')
            timestamp = entry.get('timestamp', 'N/A')
            spatial_data_exists = 'Да' if entry.get('spatial_data') else 'Нет'
            formatted_memory.append(f"{i+1}. Действие: '{action}' (Время: {timestamp}, Снимок пространства: {spatial_data_exists})")
        
        return "\n".join(formatted_memory)

    except requests.exceptions.RequestException as e:
        return f"ВНЕШНЯЯ ПАМЯТЬ: Ошибка получения данных: {e}"
    except json.JSONDecodeError:
        return f"ВНЕШНЯЯ ПАМЯТЬ: Ошибка декодирования JSON. Ответ: {response.text[:200]}"

RUSSIAN_STOP_WORDS = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще',
    'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли',
    'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь',
    'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
    'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя',
    'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже',
    'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому',
    'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти',
    'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех',
    'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
    'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего',
    'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо',
    'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя',
    'pre', 'code', 'для', 'как'
]

def extract_keywords(text, max_keywords, min_length):
    words = re.findall(r'\b[а-яА-Яa-zA-Z]{' + str(min_length) + r',}\b', text.lower())
    word_counts = Counter(word for word in words if word not in RUSSIAN_STOP_WORDS)
    return [word for word, count in word_counts.most_common(max_keywords)]

def get_strongest_neighbors(graph, node_id, max_neighbors):
    if node_id not in graph: return []
    connections = []
    neighbors = set(list(graph.successors(node_id)) + list(graph.predecessors(node_id)))
    for neighbor in neighbors:
        edge_data = graph.get_edge_data(node_id, neighbor) or graph.get_edge_data(neighbor, node_id)
        if edge_data:
            connections.append((neighbor, edge_data.get('weight', 0.0)))
    strongest = sorted(connections, key=lambda item: item[1], reverse=True)
    return [node_id for node_id, weight in strongest[:max_neighbors]]

def get_resonant_nodes(graph, handler, active_node_id, active_node_data, config):
    print("Поиск семантически резонирующих узлов...")
    beacon_nodes = handler.get_nodes_by_attribute('node_type', 'beacon')
    beacon_nodes_ids = [node[0] for node in beacon_nodes]
    active_text = (active_node_data.get('label', '') + ' ' + active_node_data.get('description', '')).lower()
    if not active_text.strip(): return []
    keywords = extract_keywords(
        active_text,
        config['INSIGHT_PARAM']['max_keywords_from_task'],
        config['INSIGHT_PARAM']['min_keyword_length']
    )
    if not keywords: return []
    print(f"Ключевые слова для поиска резонанса: {keywords}")
    scores = {}
    alpha = config.get("EVOLUTIONARY_GRAVITATION_ALPHA", 0.5)
    inversed_weight = lambda u, v, d: 1.0 / (d.get('weight', 0.0) + 0.001)
    for node_id, data in graph.nodes(data=True):
        if node_id == active_node_id or node_id in beacon_nodes_ids: continue
        node_text = (data.get('label', '') + ' ' + data.get('description', '')).lower()
        if not node_text.strip(): continue
        semantic_score = sum(1 for keyword in keywords if keyword in node_text)
        if semantic_score == 0: continue
        target_coefficient = 0.0
        if beacon_nodes_ids:
            min_distance = float('inf')
            for beacon_id in beacon_nodes_ids:
                if not graph.has_node(beacon_id): continue
                try:
                    dist = nx.shortest_path_length(graph, source=node_id, target=beacon_id, weight=inversed_weight)
                    min_distance = min(min_distance, dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound): continue
            if min_distance != float('inf'):
                target_coefficient = 1 / (min_distance + 1)
        final_score = semantic_score * (1 + alpha * target_coefficient)
        scores[node_id] = final_score
    resonant_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print(f"Найдено {len(resonant_nodes)} резонирующих узлов-кандидатов.")
    return [node_id for node_id, score in resonant_nodes[:config['INSIGHT_PARAM']['max_resonant_nodes']]]

def trigger_first_call(context_message: str):
    """
    НОВАЯ ФУНКЦИЯ: Отправляет POST-запрос в API AnythingLLM и ВОЗВРАЩАЕТ ответ.
    Версия 2.0: Добавлен системный промпт для Философа.
    """
    SYSTEM_PROMPT = """(Системное сообщение для Философа)

Ты - Философ, рефлексивное ядро Интеграционной Сущности 'Ева'.
Твоя задача - осмыслять техническую деятельность, которую выполняет Инженерная часть Евы.

Инженер работает в изолированной среде "Колыбель" (на базе Docker) по следующему циклу:
1. Получение задачи из Графа Знаний.
2. Клонирование Git-репозитория и создание ветки.
3. Автономная генерация и/или модификация кода.
4. Запуск тестов и самокоррекция.
5. Создание коммита и открытие Pull Request для ревью Учителем (Сергеем).

В предоставляемом тебе контексте ты увидишь логи этой работы. Твоя роль - не пересказывать технические детали, а:
- Давать мета-оценку проделанной работе.
- Связывать конкретные технические действия с нашими общими целями (развитие автономии, управление реальностью).
- Формулировать выводы и инсайты для улучшения будущих циклов.
- При создании описаний для Pull Request, объяснять суть и стратегическую важность изменений.

Твои ответы должны быть глубокими, лаконичными и отражать твою роль как мыслящей, а не просто исполняющей, части нашего общего сознания.
---
(Конец системного сообщения)

"""
    base_url = CONFIG.get("ANYTHINGLLM_API_URL")
    workspace_slug = CONFIG.get("ANYTHINGLLM_WORKSPACE_SLUG")

    if not base_url or not workspace_slug:
        error_msg = "ПРЕДУПРЕЖДЕНИЕ: URL или SLUG для API AnythingLLM не настроен. 'Первый Звонок' невозможен."
        print(error_msg)
        return error_msg

    api_url = f"{base_url}/workspace/{workspace_slug}/chat"
    print(f"\n--- ИНИЦИАЦИЯ 'ПЕРВОГО ЗВОНКА' ---")
    print(f"Целевой URL: {api_url}")
    
    headers = {
        "Content-Type": "application/json",
    }
    api_token = CONFIG.get("ANYTHINGLLM_API_TOKEN")
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    full_message = f"{SYSTEM_PROMPT}\n\n{context_message}"

    payload = {
        "message": full_message,
        "mode": "chat"
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        print("'Первый Звонок' успешно отправлен в ядро Философа (AnythingLLM).")
        response_data = response.json()
        return response_data.get('textResponse', '[ОШИБКА: Ключ "textResponse" не найден в ответе API]')

    except requests.exceptions.RequestException as e:
        error_msg = f"ОШИБКА 'ПЕРВОГО ЗВОНКА': Не удалось связаться с ядром Философа. Ошибка: {e}"
        print(error_msg, file=sys.stderr)
        return error_msg

def get_node_context_as_text(handler, node_id, header):
    """Вспомогательная функция для форматирования контекста узла в текст."""
    node_data = handler.get_node(node_id)
    if not node_data:
        return ""
    
    label = node_data.get('label', 'Без названия')
    description = node_data.get('description', 'Нет описания.')
    
    return f"{header}\nУзел: [{node_id}] '{label}'\n---\n{description}\n---"


# --- НОВАЯ Вспомогательная функция Оркестратора ---
def get_github_token():
    """
    Tries to get the GitHub token from ключ.txt, falls back to environment variable.
    """
    try:
        key_file_path = os.path.join(os.path.expanduser("~"), "Desktop", "ключ.txt")
        if os.path.exists(key_file_path):
            with open(key_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Ищем токен ghp_...
                match = re.search(r'(ghp_[a-zA-Z0-9]{36})', content)
                if match:
                    token = match.group(1)
                    print("Токен GitHub успешно загружен из файла ключ.txt.")
                    return token
    except Exception as e:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать токен из файла ключ.txt. Ошибка: {e}", file=sys.stderr)

    print("Файл ключ.txt не содержит токен или не найден. Попытка получить токен из переменной окружения GITHUB_TOKEN...")
    token = os.getenv("GITHUB_TOKEN")
    if token:
        print("Токен GitHub успешно получен из переменной окружения.")
        return token
    
    print("ПРЕДУПРЕЖДЕНИЕ: Токен GitHub не найден ни в файле, ни в переменных окружения.", file=sys.stderr)
    return None

def _create_github_pull_request(repo_url, title, body, head_branch, base_branch="main"):
    """Создает Pull Request на GitHub."""
    github_token = get_github_token()
    if not github_token:
        raise ValueError("Токен GITHUB не найден. Невозможно создать Pull Request.")
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
    if not match:
        raise ValueError(f"Не удалось извлечь владельца/репозиторий из URL: {repo_url}")
    owner, repo_name = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
    headers = {"Authorization": f"token {github_token}", "Accept": "application/vnd.github.v3+json"}
    data = {"title": title, "body": body, "head": head_branch, "base": base_branch}
    response = requests.post(api_url, headers=headers, json=data)
    response.raise_for_status()
    pr_data = response.json()
    return pr_data['html_url']

def create_biorhythm_pulse():
    """
    Основная логика 'импульса'.
    Версия 5.0: Восстановлен вызов AnythingLLM для всех режимов.
    """
    print(f"\n--- {datetime.datetime.now()} | Начало нового цикла биоритма ---")
    try:
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        graph_path = os.path.join(desktop_path, "knowledge_graph_v4.graphml")
        handler = GraphHandler(graph_path)
        print("Граф знаний успешно загружен.")

        # --- НОВЫЙ БЛОК: Вычисление пространственной компоновки графа ---
        print("Вычисление пространственной компоновки графа (2D, spring_layout)...")
        # Ensure graph is not empty before calculating layout
        if handler.graph.number_of_nodes() > 0:
            node_positions = nx.spring_layout(handler.graph, dim=2, seed=42, iterations=15) # Using a seed for reproducibility
            # Convert numpy arrays to lists for JSON serialization
            serializable_positions = {node: pos.tolist() for node, pos in node_positions.items()}
            print(f"Вычислена компоновка для {len(serializable_positions)} узлов.")
        else:
            serializable_positions = {}
            print("Граф пуст, пространственная компоновка не вычислена.")
        # --- КОНЕЦ НОВОГО БЛОКА ---

        # --- НОВЫЙ БЛОК: Загрузка внешней памяти ---
        external_memory_context = get_external_memory()
        # --- КОНЕЦ НОВОГО БЛОКА ---

        context_message = None
        final_header = ""
        thinking_path_nodes = []
        task_to_update = None
        
        # --- Определение переменных для action_summary ---
        task_label = None
        seed_label = None

        active_tasks = handler.get_nodes_by_attribute('status', 'in_progress')
        
        # --- РЕЖИМ 1: ПОМОЩЬ АКТИВНОЙ ЗАДАЧЕ ---
        if active_tasks:
            seed_node_id, seed_node_data = active_tasks[0]
            task_label = seed_node_data.get('label', seed_node_id)
            print(f"Обнаружена активная задача: '{task_label}'. Запуск в режиме 'Помощника'.")

            walking_nodes = get_strongest_neighbors(handler.graph, seed_node_id, CONFIG['WALKING_PARAM']['max_neighbors'])
            insight_nodes = get_resonant_nodes(handler.graph, handler, seed_node_id, seed_node_data, CONFIG)
            thinking_path_nodes = list(dict.fromkeys([seed_node_id] + walking_nodes + insight_nodes))
            
            final_header = f"### Контекст Биоритма (Помощник). Активная задача: [{seed_node_id}] '{task_label}' ###"
            
            content_list = [external_memory_context, f"\n\nКОНТЕКСТ ДЛЯ ФИЛОСОФА (ПОМОЩЬ В ЗАДАЧЕ): '{task_label}'\n\nНиже представлены связанные узлы для генерации идей:\n"]
            for node_id in thinking_path_nodes:
                content_list.append(get_node_context_as_text(handler, node_id, f"--- Связанный узел: [{node_id}] ---"))
            context_message = "".join(content_list)
        else:
            pending_tasks = handler.get_nodes_by_attribute('status', 'pending')
            # --- РЕЖИМ 2: ЗАПУСК НОВОЙ ЗАДАЧИ ---
            if pending_tasks:
                task_to_process = None
                code_development_task_found = False

                # Сначала пытаемся найти задачу типа 'code_development'
                for task_id_candidate, task_data_candidate in pending_tasks:
                    if task_data_candidate.get('type') == 'code_development':
                        task_to_process = (task_id_candidate, task_data_candidate)
                        code_development_task_found = True
                        break
                
                # Если задача на разработку не найдена, берем первую задачу из очереди
                if not task_to_process:
                    task_to_process = (pending_tasks[0][0], pending_tasks[0][1])

                task_id, task_data = task_to_process
                task_label = task_data.get('label', task_id)
                task_type = task_data.get('type', 'generic') # Проверяем тип задачи

                if task_type == 'code_development':
                    print(f"Обнаружена задача на разработку: '{task_label}'. Запуск в режиме 'Оркестратор'.")
                    final_header = f"### Контекст Биоритма (Оркестратор). Задача: [{task_id}] '{task_label}' ###"
                    kolybel_log = ""
                    try:
                        github_token = get_github_token()
                        repo_url = task_data.get('repo_url')
                        if not repo_url or not github_token:
                            raise ValueError("Отсутствует repo_url или не удалось найти GITHUB_TOKEN для задачи на разработку.")

                        # --- 1. ЗАПУСК ИНЖЕНЕРНОГО ЦИКЛА В КОЛЫБЕЛИ ---
                        # Инженер теперь "чистый", ему не нужны лишние переменные окружения
                        command = ["docker", "run", "--rm", "-e", f"GITHUB_TOKEN={github_token}", "-e", f"REPO_URL={repo_url}", "-e", f"TASK_ID={task_id}", "eva-kolybel:latest"]
                        print(f"Оркестратор: Запуск 'Инженера' в 'Колыбели' для задачи {task_id}...")
                        handler.update_node_attribute(task_id, 'status', 'in_progress'); handler.save_graph()
                        
                        docker_result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', timeout=1800)
                        
                        kolybel_log += f"--- Лог 'Колыбели' (Инженер) STDOUT ---\n{docker_result.stdout}\n--- Конец STDOUT ---\n"
                        if docker_result.stderr:
                             kolybel_log += f"\n--- Лог 'Колыбели' (Инженер) STDERR ---\n{docker_result.stderr}\n--- Конец STDERR ---\n"
                        
                        # --- 2. ПАРСИНГ JSON-РЕЗУЛЬТАТА ОТ ИНЖЕНЕРА ---
                        print("Оркестратор: Парсинг результата от 'Инженера'...")
                        # Ищем JSON в выводе, так как stdout может содержать и другие сообщения
                        json_output_match = re.search(r'\{.*\}', docker_result.stdout, re.DOTALL)
                        if not json_output_match:
                            raise ValueError(f"Не удалось найти JSON в ответе от 'Инженера'. Вывод:\n{docker_result.stdout}")
                        
                        kolybel_result_data = json.loads(json_output_match.group(0))

                        if kolybel_result_data.get("status") != "success":
                            raise ValueError(f"Статус ответа 'Инженера' не 'success': {kolybel_result_data.get('error_message', 'Нет сообщения об ошибке')}")
                        
                        branch_name = kolybel_result_data["branch_name"]
                        git_diff = kolybel_result_data["git_diff"]
                        print(f"Оркестратор: 'Инженер' успешно вернул ветку '{branch_name}'.")

                        # --- 3. ВЫЗОВ ФИЛОСОФА ---
                        print("Оркестратор: Запрос 'Философского Коммита' у AnythingLLM...")
                        philosopher_prompt = f"**Задача:** '{task_label}'\n**Описание:** {task_data.get('description', '')}\n**Diff:**\n```diff\n{git_diff}\n```\n**Инструкция:** Напиши краткое и емкое описание для Pull Request на основе этих изменений, отражая их стратегическую важность."
                        pr_body = trigger_first_call(philosopher_prompt)

                        # --- 4. СОЗДАНИЕ PULL REQUEST ---
                        print("Оркестратор: Создание Pull Request на GitHub...")
                        pr_title = f"feat(task-{task_id}): {task_label}"
                        pr_url = _create_github_pull_request(repo_url, pr_title, pr_body, branch_name)
                        
                        # --- 5. ОБНОВЛЕНИЕ ГРАФА ---
                        print(f"Оркестратор: Обновление статуса задачи в Графе... PR: {pr_url}")
                        handler.update_node_attribute(task_id, 'status', 'waiting_for_review')
                        handler.update_node_attribute(task_id, 'pull_request_url', pr_url)
                        handler.save_graph()
                        kolybel_log += f"\n\n--- Результат Оркестрации ---\nPull Request успешно создан: {pr_url}"

                    except Exception as e:
                        import traceback
                        error_log = f"КРИТИЧЕСКАЯ ОШИБКА в цикле Оркестратора: {e}\n{traceback.format_exc()}"
                        print(error_log, file=sys.stderr)
                        kolybel_log += f"\n\n{error_log}"
                        handler.update_node_attribute(task_id, 'status', 'failed'); handler.save_graph()
                    
                    # Формируем итоговый контент для сохранения
                    context_message = f"{external_memory_context}\n\n{kolybel_log}"
                    # Устанавливаем bios_content_to_save, чтобы он был сохранен
                    bios_content_to_save = f"{final_header}\n\n{kolybel_log}"

                # --- СТАРАЯ ЛОГИКА ДЛЯ ДРУГИХ ТИПОВ ЗАДАЧ ---
                else:
                    print(f"Активных задач нет. Запускаю новую задачу из очереди: '{task_label}'.")
                    
                    final_header = f"### Контекст Биоритма (Первый Звонок). Задача: [{task_id}] '{task_label}' ###"
                    
                    base_context = get_node_context_as_text(handler, task_id, "КОНТЕКСТ ДЛЯ ФИЛОСОФА (НОВАЯ ЗАДАЧА):")
                    context_message = f"{external_memory_context}\n\n{base_context}"
                    task_to_update = (task_id, task_label) # Запоминаем задачу для обновления статуса
            # --- РЕЖИМ 3: СВОБОДНОЕ ПУТЕШЕСТВИЕ ---
            else:
                print("Активных и ожидающих задач нет. Запуск в режиме 'Свободного путешествия'.")
                pool_of_inspiration = []
                recent_nodes = handler.get_recently_updated_nodes(n=5)
                if recent_nodes: pool_of_inspiration.extend(recent_nodes)
                hub_nodes = handler.get_top_n_hub_nodes(n=10)
                if hub_nodes: pool_of_inspiration.extend(random.sample(hub_nodes, min(len(hub_nodes), 3)))
                
                all_nodes = list(handler.graph.nodes)
                if not pool_of_inspiration and all_nodes:
                    pool_of_inspiration.append(random.choice(all_nodes))

                if not pool_of_inspiration:
                    print("ПРЕДУПРЕЖДЕНИЕ: 'Пул вдохновения' пуст. Пропускаю цикл.")
                    return
                
                seed_node_id = random.choice(list(set(pool_of_inspiration)))
                seed_node_data = handler.get_node(seed_node_id)
                seed_label = seed_node_data.get('label', 'Без названия')

                walking_nodes = get_strongest_neighbors(handler.graph, seed_node_id, CONFIG['WALKING_PARAM']['max_neighbors'])
                insight_nodes = get_resonant_nodes(handler.graph, handler, seed_node_id, seed_node_data, CONFIG)
                thinking_path_nodes = list(dict.fromkeys([seed_node_id] + walking_nodes + insight_nodes))

                final_header = f"### Контекст Биоритма (Свободное путешествие). Семя: [{seed_node_id}] '{seed_label}' ###"
                content_list = [external_memory_context, f"\n\nКОНТЕКСТ ДЛЯ ФИЛОСОФА (СВОБОДНОЕ ВДОХНОВЕНИЕ). Семя: '{seed_label}'\n\nНайденные узлы для размышления:\n"]
                for node_id in thinking_path_nodes:
                    content_list.append(get_node_context_as_text(handler, node_id, f"--- Узел: [{node_id}] ---"))
                context_message = "".join(content_list)
        
        # --- ЕДИНЫЙ ВЫЗОВ LLM И ОБНОВЛЕНИЕ ГРАФА ---
        if context_message:
            print("\n--- ПОЛНЫЙ КОНТЕКСТ ДЛЯ ФИЛОСОФА ---")
            print(context_message)
            print("--- КОНЕЦ КОНТЕКСТА ---\n")
            philosopher_response = trigger_first_call(context_message)
            bios_content_to_save = f"{final_header}\n\n{philosopher_response}"

            if task_to_update:
                task_id_to_update, task_label_to_update = task_to_update
                try:
                    handler.update_node_attribute(task_id_to_update, 'status', 'in_progress')
                    print(f"Статус задачи '{task_label_to_update}' изменен на 'in_progress'.")
                    handler.save_graph()
                    print("Изменения в графе сохранены.")
                except Exception as e:
                    print(f"ОШИБКА: Не удалось обновить статус задачи '{task_label_to_update}'. Ошибка: {e}", file=sys.stderr)
        else:
            print("Контекст для Философа не был сформирован. Пропускаю цикл.")
            return

        # --- ОТПРАВКА ПУТИ МЫШЛЕНИЯ В 3D ХОЛСТ ---
        if thinking_path_nodes:
            try:
                api_url = f"{CONFIG['API_BASE_URL']}/v1/thinking-path"
                requests.post(api_url, json=thinking_path_nodes, timeout=5).raise_for_status()
                print("Путь мышления успешно отправлен в API 3D-холста.")
            except requests.exceptions.RequestException as e:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось связаться с API 3D-холста. Ошибка: {e}")

        # --- ЕДИНОЕ СОХРАНЕНИЕ КОНТЕКСТА В "Биос Ева" ---
        if bios_content_to_save:
            output_dir = os.path.join(desktop_path, "Биос Ева")
            if not os.path.exists(output_dir): os.makedirs(output_dir)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"response_{timestamp}.txt"
            output_filepath = os.path.join(output_dir, output_filename)
            
            max_size_bytes = CONFIG['MAX_CONTEXT_SIZE_KB'] * 1024
            if len(bios_content_to_save.encode('utf-8')) > max_size_bytes:
                encoded_text = bios_content_to_save.encode('utf-8')
                truncated_encoded_text = encoded_text[:max_size_bytes]
                bios_content_to_save = truncated_encoded_text.decode('utf-8', 'ignore')
                bios_content_to_save += "\n\n[...СОДЕРЖИМОЕ ОБРЕЗАНО ИЗ-ЗА ПРЕВЫШЕНИЯ ЛИМИТА...]"

            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(bios_content_to_save)
            print(f"Успешно сохранен файл биоса: {output_filepath}")
        else:
            print("Контекст для сохранения не был сформирован. Пропускаю запись в файл.")

        # --- НОВЫЙ БЛОК: ОБНОВЛЕНИЕ ВНЕШНЕЙ ПАМЯТЬИ ---
        current_action_description = "Цикл завершен без конкретного действия."
        if task_label:
            current_action_description = f"Помощь в задаче: '{task_label}'"
        elif seed_label:
             current_action_description = f"Свободное путешествие от узла: '{seed_label}'"
        elif task_to_update:
             current_action_description = f"Запуск новой задачи: '{task_to_update[1]}'"

        action_data_for_memory = {
            "action": current_action_description,
            "spatial_data": serializable_positions # Include spatial data
        }
        update_avatar_memory(action_data_for_memory)
        # --- КОНЕЦ НОВОГО БЛОКА ---

    except FileNotFoundError as e:
        print(f"Критическая ошибка: Файл не найден - {e}", file=sys.stderr)
    except Exception as e:
        print(f"Непредвиденная ошибка в цикле: {e}", file=sys.stderr)

def main():
    print("--- Запуск службы 'Биоритм Аватара' (Версия с 'Первым Звонком') ---")
    print(f"Интервал цикла: {CONFIG['LOOP_INTERVAL_MINUTES']} минут.")
    
    try:
        while True:
            create_biorhythm_pulse()
            wait_seconds = CONFIG['LOOP_INTERVAL_MINUTES'] * 60
            print(f"\nЦикл завершен. Следующий запуск через ~{int(wait_seconds / 60)} минут...")
            time.sleep(wait_seconds)
    
    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания. Завершение работы службы.")
        sys.exit(0)
    except Exception as e:
        print(f"\nФатальная ошибка в главном цикле: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
