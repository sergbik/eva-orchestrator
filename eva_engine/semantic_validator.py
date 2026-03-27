# -*- coding: utf-8 -*-
import ast
import sys

# Правила для валидации, вынесены для наглядности
RULES = {
    'DISALLOWED_CALLS': {
        'handler.get_all_nodes': 'Обнаружен запрещенный вызов handler.get_all_nodes(). Используйте get_nodes_by_attribute для получения узлов по категориям.',
    },
    'MODIFICATION_FUNCTIONS': [
        'handler.add_node',
        'handler.update_node_attribute',
        'handler.add_edge',
        'handler.delete_node',
        'handler.delete_edge'
    ],
    'SAVE_FUNCTION': 'handler.save_graph',
    'WRITE_FUNCTION': 'write_file_content'
}

class CodeValidator(ast.NodeVisitor):
    """
    Класс для обхода AST и поиска потенциальных проблем в коде.
    """
    def __init__(self):
        self.findings = []
        self.modifies_graph = False
        self.saves_graph = False

    def visit_Call(self, node):
        call_name = self._get_full_call_name(node.func)
        if not call_name:
            self.generic_visit(node)
            return

        # Проверка 1: Запрещенные вызовы
        if call_name in RULES['DISALLOWED_CALLS']:
            self.findings.append(f"[Строка {node.lineno}] {RULES['DISALLOWED_CALLS'][call_name]}")

        # Проверка 2: Отсутствие сохранения после модификации
        if call_name in RULES['MODIFICATION_FUNCTIONS']:
            self.modifies_graph = True
        if call_name == RULES['SAVE_FUNCTION']:
            self.saves_graph = True
            
        # Проверка 3: Небезопасная запись в системные файлы
        if call_name == RULES['WRITE_FUNCTION']:
            if len(node.args) > 0:
                arg_value = self._get_arg_value(node.args[0])
                # Проверяем, что путь к файлу - строка и что он указывает на системный файл
                if isinstance(arg_value, str) and (arg_value.endswith('.py') or arg_value.endswith('.html')) and 'код.txt' not in arg_value:
                    self.findings.append(f"[Строка {node.lineno}] Обнаружена потенциально небезопасная запись в системный файл '{arg_value}'. Для изменения кода должен использоваться 'Протокол Безопасной Модернизации'.")

        self.generic_visit(node)

    def _get_full_call_name(self, node):
        if isinstance(node, ast.Attribute):
            parent = self._get_full_call_name(node.value)
            return f"{parent}.{node.attr}" if parent else None
        elif isinstance(node, ast.Name):
            return node.id
        return None
        
    def _get_arg_value(self, arg):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str): # Python 3.8+
            return arg.value
        elif isinstance(arg, ast.Str): # Python 3.7-
             return arg.s
        return None

    def report(self):
        # Итоговая проверка после обхода всего дерева
        if self.modifies_graph and not self.saves_graph:
            self.findings.append("Обнаружены операции модификации графа, но отсутствует вызов handler.save_graph() для сохранения изменений.")
        return self.findings

def validate_code(source_code: str) -> list:
    """
    Основная функция-валидатор. Принимает исходный код в виде строки.
    Возвращает список найденных проблем (строк) или пустой список, если проблем нет.
    """
    try:
        tree = ast.parse(source_code)
        validator = CodeValidator()
        validator.visit(tree)
        return validator.report()
    except SyntaxError as e:
        return [f"Критическая ошибка синтаксиса на строке {e.lineno}: {e.msg}"]
    except Exception as e:
        return [f"Непредвиденная ошибка во время валидации: {e}"]

# Этот блок позволяет запускать файл как отдельный скрипт для отладки
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Использование: python semantic_validator.py <путь_к_файлу_для_анализа.py>")
        sys.exit(1)
    
    file_to_analyze = sys.argv[1]
    try:
        with open(file_to_analyze, 'r', encoding='utf-8') as f:
            code = f.read()
        
        findings = validate_code(code)
        
        if not findings:
            print("Семантический анализ кода успешно завершен. Потенциальных проблем не обнаружено.")
        else:
            print("Обнаружены следующие потенциальные проблемы:")
            for finding in findings:
                print(f"- {finding}")

    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {file_to_analyze}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
