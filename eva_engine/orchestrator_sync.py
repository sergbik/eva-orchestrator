# -*- coding: utf-8 -*-
"""
Модуль Синхронизации Оркестратора (Нервная система Я64)
Версия: 1.2 "Solid Merge Edition"
Обеспечивает БЕЗОПАСНУЮ синхронизацию Графа Знаний без повреждения XML.
"""
import subprocess
import os
import time

class OrchestratorSync:
    def __init__(self, repo_path):
        self.repo_path = repo_path

    def _run_git(self, args):
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                encoding='utf-8',
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr

    def commit_local(self, message="Local capture"):
        """Принудительная фиксация локальных изменений."""
        self._run_git(["add", "."])
        success, msg = self._run_git(["status"])
        if "nothing to commit" not in msg:
            return self._run_git(["commit", "-m", message])
        return True, "Nothing to commit"

    def pull_memory(self):
        """Безопасная загрузка из облака. В случае конфликта XML - выбираем локальную версию."""
        print(f"[{time.ctime()}] Синхронизация: Pulling memory (Safety Mode)...")
        # Используем стратегию 'ours', чтобы Git не вставлял маркеры в XML
        return self._run_git(["pull", "--rebase", "-X", "ours", "origin", "main"])

    def push_memory(self, commit_message="Pulse sync"):
        """Отправка изменений в облако."""
        self._run_git(["config", "user.name", "EVA2^2^8 Cloud"])
        self._run_git(["config", "user.email", "eva-cloud@ya64.pro"])
        self._run_git(["add", "."])
        success_check, msg_check = self._run_git(["status"])
        if "nothing to commit" in msg_check:
            return True, "Nothing to commit"
        success, msg = self._run_git(["commit", "-m", commit_message])
        if success:
            return self._run_git(["push", "origin", "main"])
        return False, msg

if __name__ == "__main__":
    print("OrchestratorSync v1.2 ready.")
