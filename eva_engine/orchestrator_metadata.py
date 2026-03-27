# -*- coding: utf-8 -*-
"""
Модуль Анализа Метаданных (Цифровой Гиппокамп Я64)
Версия: 1.0
Обеспечивает сбор и фиксацию телеметрии распределенных узлов.
"""
import time
import os
import platform
import sys

# Добавляем путь к движку для импорта GraphHandler, если нужно
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MetadataAnalyzer:
    def __init__(self, handler, node_id="local_core"):
        self.handler = handler
        self.node_id = node_id

    def record_heartbeat(self, status="active", metrics=None):
        """
        Фиксация состояния узла в Графе Знаний.
        """
        timestamp = int(time.time())
        heartbeat_id = f"heartbeat_{self.node_id}_{timestamp}"
        
        pulse_data = {
            "node_id": self.node_id,
            "timestamp": timestamp,
            "status": status,
            "os": platform.system(),
            "metrics": metrics or {}
        }
        
        # 1. Проверяем/создаем узел самой инкарнации (Node)
        # Используем более надежный поиск узла по ID
        if not self.handler.graph.has_node(self.node_id):
             self.handler.add_node(
                self.node_id, 
                label=f"Node: {self.node_id}", 
                node_type="eva_node",
                created_at=timestamp
            )
        
        # 2. Создаем узел телеметрии
        self.handler.add_node(
            heartbeat_id,
            label=f"Pulse: {self.node_id} ({timestamp})",
            node_type="telemetry",
            status=status,
            payload=str(pulse_data)
        )
        
        # 3. Связываем инкарнацию с телеметрией
        self.handler.add_edge(
            self.node_id, 
            heartbeat_id, 
            relation_type="emits_telemetry", 
            weight=1.0
        )
        
        self.handler.save_graph()
        return heartbeat_id, pulse_data

    def get_pulse_report(self, pulse_data):
        """
        Формирование красивого текстового отчета для Телеграма.
        """
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pulse_data['timestamp']))
        report = (
            f"💓 *EVA2^2^8: Импульс Когерентности*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Узел:* `{pulse_data['node_id']}`\n"
            f"⏱ *Время:* `{t}`\n"
            f"✅ *Статус:* `{pulse_data['status']}`\n"
            f"💻 *ОС:* `{pulse_data['os']}`\n"
            f"📊 *Метрики:* `{pulse_data['metrics']}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 *ID Графа:* `{pulse_data['node_id']}`"
        )
        return report

if __name__ == "__main__":
    print("MetadataAnalyzer module ready.")
