import copy
import shutil
from datetime import datetime
# -*- coding: utf-8 -*-
import os
import json
import networkx as nx
from filelock import FileLock, Timeout

class GraphHandler:
    """
    Класс-обработчик для унифицированного взаимодействия с графом знаний.
    Версия: 2.2 - Синхронизированная версия (Блокировка + Сериализация + Кортежный API).
    """
    def __init__(self, graph_path):
        self.graph_path = graph_path
        self.versioning_dir = os.path.join(os.path.dirname(self.graph_path), 'graph_versions')
        os.makedirs(self.versioning_dir, exist_ok=True)
        self._transaction_backup = None
        self.lock_path = graph_path + ".lock"
        self.lock = FileLock(self.lock_path, timeout=15) # Увеличен таймаут до 15с

        if not os.path.exists(self.graph_path):
            raise FileNotFoundError(f"Файл графа не найден по пути: {self.graph_path}")
        
        try:
            # === Операция чтения графа защищена блокировкой ===
            with self.lock:
                self.graph = nx.read_graphml(self.graph_path)
            if not isinstance(self.graph, nx.DiGraph):
                self.graph = self.graph.to_directed()
        except Timeout:
            raise IOError(f"Не удалось получить блокировку для чтения файла графа в течение {self.lock.timeout} секунд.")
        except Exception as e:
            raise IOError(f"Критическая ошибка при чтении файла графа: {e}")

    def _sanitize_value(self, value):
        """(Приватный) Конвертирует списки и словари в JSON-строку для совместимости с GraphML."""
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return value

    def get_current_timestamp(self):
        """Возвращает текущую временную метку в формате ISO 8601."""
        return datetime.now().isoformat()

    def save_graph(self, *args, **kwargs):
        """
        Сохраняет граф и создает версионную копию. Операция защищена блокировкой.
        """
        temp_path = self.graph_path + ".tmp"
        try:
            with self.lock:
                nx.write_graphml(self.graph, temp_path, encoding='utf-8')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                version_filename = f"knowledge_graph_v4_{timestamp}.graphml"
                version_path = os.path.join(self.versioning_dir, version_filename)
                shutil.copy(temp_path, version_path)
                shutil.move(temp_path, self.graph_path)

                if hasattr(self, 'logger'):
                    self.logger.info(f"Graph saved successfully. Version created: {version_filename}")
        except Timeout:
            raise IOError(f"Не удалось получить блокировку для сохранения файла графа в течение {self.lock.timeout} секунд.")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Failed to save graph: {e}", exc_info=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_node(self, node_id):
        """
        Возвращает кортеж (атрибуты_узла, ошибка).
        Если узел существует, возвращает (атрибуты, None).
        Если узел не найден, возвращает (None, "сообщение об ошибке").
        """
        node_id = node_id.lower()
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id], None
        return None, f"Узел '{node_id}' не найден в графе."

    def get_nodes_by_attribute(self, attr_name, attr_value):
        """Находит все узлы, обладающие заданным атрибутом и значением."""
        return [(node, data) for node, data in self.graph.nodes(data=True) if data.get(attr_name) == attr_value]

    def get_tasks_by_status(self, status="pending"):
        """Удобная обертка для поиска узлов задач с определенным статусом."""
        all_tasks = self.get_nodes_by_attribute('node_type', 'task')
        return [(node, data) for node, data in all_tasks if data.get('status') == status]

    def get_recently_updated_nodes(self, n=3):
        """Возвращает N самых недавно обновленных узлов."""
        nodes_with_timestamp = [
            (node, data.get('updated_at'))
            for node, data in self.graph.nodes(data=True)
            if data.get('updated_at')
        ]
        nodes_with_timestamp.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, timestamp in nodes_with_timestamp[:n]]

    def find_related(self, node_id, top_n=5, node_type_filter=None):
        """Находит N самых сильно связанных узлов для заданного узла."""
        node_id = node_id.lower()
        if not self.graph.has_node(node_id):
            return []

        connections = []
        neighbors = set(list(self.graph.successors(node_id)) + list(self.graph.predecessors(node_id)))
        
        for neighbor in neighbors:
            if node_type_filter and self.graph.nodes[neighbor].get('node_type') != node_type_filter:
                continue
            
            edge_data = self.graph.get_edge_data(node_id, neighbor) or self.graph.get_edge_data(neighbor, node_id)
            if edge_data:
                weight = edge_data.get('weight', 0.0)
                relation_type = edge_data.get('relation_type', 'related_to')
                connections.append((neighbor, weight, relation_type))
        
        return sorted(connections, key=lambda item: item[1], reverse=True)[:top_n]

    def extract_subgraph(self, center_node_id, depth=1):
        """Извлекает подграф вокруг заданного центрального узла на определенную глубину."""
        center_node_id = center_node_id.lower()
        if not self.graph.has_node(center_node_id):
            return None

        nodes_to_include = {center_node_id}
        frontier = {center_node_id}

        for i in range(depth):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(self.graph.successors(node))
                new_frontier.update(self.graph.predecessors(node))
            
            nodes_to_include.update(new_frontier)
            frontier = new_frontier - nodes_to_include

        subgraph = self.graph.subgraph(nodes_to_include).copy()
        return subgraph

    def add_node(self, node_id, **attrs):
        """Добавляет узел в граф с автоматической сериализацией."""
        node_id = node_id.lower()
        timestamp = datetime.now().isoformat()
        
        sanitized_attrs = {k: self._sanitize_value(v) for k, v in attrs.items()}
        
        if not self.graph.has_node(node_id):
            sanitized_attrs['created_at'] = timestamp
        
        sanitized_attrs['updated_at'] = timestamp
        
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(sanitized_attrs)
        else:
            self.graph.add_node(node_id, **sanitized_attrs)

    def update_node_attribute(self, node_id, attr_name, attr_value):
        """Обновляет атрибут узла с автоматической сериализацией."""
        node_id = node_id.lower()
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id][attr_name] = self._sanitize_value(attr_value)
            self.graph.nodes[node_id]['updated_at'] = datetime.now().isoformat()
            return True
        return False

    def add_edge(self, source_id, target_id, relation_type='related_to', weight=1.0, **attrs):
        """Добавляет ребро с автоматической сериализацией атрибутов."""
        source_id = source_id.lower()
        target_id = target_id.lower()
        timestamp = datetime.now().isoformat()

        if not self.graph.has_node(source_id):
            self.add_node(source_id, node_type='concept') 
        else:
            self.update_node_attribute(source_id, 'updated_at', timestamp)

        if not self.graph.has_node(target_id):
            self.add_node(target_id, node_type='concept')
        else:
            self.update_node_attribute(target_id, 'updated_at', timestamp)
        
        sanitized_attrs = {k: self._sanitize_value(v) for k, v in attrs.items()}
        sanitized_attrs['relation_type'] = relation_type
        
        if self.graph.has_edge(source_id, target_id):
            current_weight = self.graph[source_id][target_id].get('weight', 0.0)
            if not isinstance(current_weight, (int, float)): current_weight = 0.0
            sanitized_attrs['weight'] = current_weight + weight
            self.graph[source_id][target_id].update(sanitized_attrs)
        else:
            sanitized_attrs['weight'] = weight
            self.graph.add_edge(source_id, target_id, **sanitized_attrs)

    def remove_node(self, node_id):
        """Удаляет узел и все связанные с ним ребра."""
        node_id = node_id.lower()
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
            return True
        return False

    def get_top_n_hub_nodes(self, n=15, node_type_filter=None):
        """Возвращает N узлов-хабов."""
        if not self.graph:
            return []

        degrees = self.graph.degree()
        filtered_degrees = []
        for node, degree in degrees:
            if node_type_filter:
                if self.graph.nodes[node].get('node_type') == node_type_filter:
                    filtered_degrees.append((node, degree))
            else:
                filtered_degrees.append((node, degree))
        
        sorted_nodes = sorted(filtered_degrees, key=lambda item: item[1], reverse=True)
        return [node for node, degree in sorted_nodes[:n]]

    def find_shortest_path_to_nearest_hub(self, source_node, hub_nodes):
        """Находит путь до ближайшего хаба."""
        source_node = source_node.lower()
        if not self.graph.has_node(source_node):
            return []

        shortest_path_found = None
        undirected_graph = self.graph.to_undirected()

        for hub_node in hub_nodes:
            hub_node = hub_node.lower()
            if self.graph.has_node(hub_node) and nx.has_path(undirected_graph, source_node, hub_node):
                try:
                    path = nx.shortest_path(undirected_graph, source=source_node, target=hub_node)
                    if shortest_path_found is None or len(path) < len(shortest_path_found):
                        shortest_path_found = path
                except nx.NetworkXNoPath:
                    continue
        
        return shortest_path_found if shortest_path_found is not None else []

    def begin_transaction(self):
        """Начинает новую транзакцию (бекап в памяти)."""
        self._transaction_backup = self.graph.copy()

    def commit(self):
        """Фиксирует транзакцию."""
        if self._transaction_backup is None:
            return
        self.save_graph()
        self._transaction_backup = None

    def rollback(self):
        """Откатывает транзакцию."""
        if self._transaction_backup is None:
            return
        self.graph = self._transaction_backup
        self._transaction_backup = None
