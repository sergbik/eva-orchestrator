# -*- coding: utf-8 -*-
"""
Bootstrap Setup: ГЕН АКТИВАЦИИ Я64
Версия: 1.0 (Supergenome)
Назначение: Автоматическое развертывание узла Евы на новом компьютере (Win/Mac/Linux).
"""
import os
import sys
import subprocess
import platform

def run_cmd(cmd):
    try:
        subprocess.run(cmd, shell=True, check=True)
        return True
    except:
        return False

def setup():
    print("--- [EVA2^2^8] ИНИЦИАЦИЯ СУПЕРГЕНОМА ---")
    print(f"Обнаружена система: {platform.system()} {platform.release()}")
    
    # 1. Проверка окружения
    print("[1/4] Проверка Python...")
    if sys.version_info < (3, 9):
        print("❌ ОШИБКА: Требуется Python 3.9 или выше.")
        return

    # 2. Установка базовых зависимостей
    print("[2/4] Установка зависимостей (networkx, requests, filelock)...")
    run_cmd("pip install networkx requests filelock google-generativeai lxml")

    # 3. Настройка памяти
    print("[3/4] Подготовка связи с Общей Памятью...")
    # Здесь можно добавить логику клонирования kolybel-workbench, если есть токен
    
    # 4. Первый запуск Оркестратора
    print("[4/4] Запуск локального ядра...")
    if os.path.exists("eva_orchestrator_core.py"):
        print("✅ Геном готов к активации.")
        # run_cmd("python eva_orchestrator_core.py")
    else:
        print("⚠️ Ядро не найдено. Убедитесь, что вы запустили скрипт в корне репозитория.")

if __name__ == "__main__":
    setup()
