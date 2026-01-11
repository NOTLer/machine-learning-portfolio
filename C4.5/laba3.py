import numpy as np
import math
from collections import Counter, defaultdict
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer

# Загрузка датасета Breast Cancer
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Данные Breast Cancer:")
print(f"Количество примеров: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Названия признаков: {feature_names}")
print(f"Метки классов: {np.unique(y)}")
print(f"Распределение классов: {np.bincount(y)}")
print(f"Класс 0: {data.target_names[0]} ({np.sum(y == 0)} примеров)")
print(f"Класс 1: {data.target_names[1]} ({np.sum(y == 1)} примеров)")
# Разделение на обучающую и тестовую выборки
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

print(f"\nРазделение данных:")
print(f"Обучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")


class C45Manual:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree = None
        self.feature_names = None

    def entropy(self, y):
        """Вычисление энтропии"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy_val = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy_val

    def information_gain(self, X, y, feature_idx, threshold):
        """Вычисление информационного выигрыша"""
        parent_entropy = self.entropy(y)

        # Разделение данных
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0

        # Энтропия после разделения
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)

        left_entropy = self.entropy(y[left_mask])
        right_entropy = self.entropy(y[right_mask])

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        return parent_entropy - child_entropy

    def gain_ratio(self, X, y, feature_idx, threshold):
        """Вычисление gain ratio (C4.5)"""
        info_gain = self.information_gain(X, y, feature_idx, threshold)

        if info_gain == 0:
            return 0

        # Вычисление split information
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold

        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            return 0

        split_info = -((n_left / n) * np.log2(n_left / n) +
                       (n_right / n) * np.log2(n_right / n))

        if split_info == 0:
            return 0

        return info_gain / split_info

    def find_best_split(self, X, y):
        """Поиск лучшего разделения"""
        best_gain_ratio = 0
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            # Получаем уникальные значения признака
            feature_values = np.unique(X[:, feature_idx])

            # Если значений слишком много, берем случайную выборку порогов для ускорения
            if len(feature_values) > 20:
                # Берем 20 случайных порогов для больших признаков
                thresholds = np.random.choice(feature_values[1:], min(20, len(feature_values) - 1), replace=False)
            else:
                # Перебираем все возможные пороги
                thresholds = [(feature_values[i] + feature_values[i + 1]) / 2
                              for i in range(len(feature_values) - 1)]

            # Перебираем возможные пороги
            for threshold in thresholds:
                gain_ratio = self.gain_ratio(X, y, feature_idx, threshold)

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain_ratio

    def build_tree(self, X, y, depth=0):
        """Рекурсивное построение дерева"""
        n_samples, n_features = X.shape

        # Условия остановки
        if (len(np.unique(y)) == 1 or
                n_samples < self.min_samples_split or
                depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]  # Возвращаем наиболее частый класс

        # Поиск лучшего разделения
        best_feature, best_threshold, best_gain_ratio = self.find_best_split(X, y)

        if best_feature is None or best_gain_ratio == 0:
            return Counter(y).most_common(1)[0][0]

        # Разделение данных
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        # Рекурсивное построение поддеревьев
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'gain_ratio': best_gain_ratio
        }

    def fit(self, X, y, feature_names=None):
        """Обучение модели"""
        self.feature_names = feature_names
        self.tree = self.build_tree(X, y)
        return self

    def predict_single(self, x, tree):
        """Предсказание для одного примера"""
        if not isinstance(tree, dict):
            return tree

        if x[tree['feature']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])

    def predict(self, X):
        """Предсказание для набора данных"""
        return np.array([self.predict_single(x, self.tree) for x in X])

    def print_tree(self, tree=None, indent="", feature_names=None):
        """Печать дерева в читаемом формате"""
        if feature_names is None:
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

        if tree is None:
            tree = self.tree

        if not isinstance(tree, dict):
            class_name = "Malignant" if tree == 0 else "Benign" if tree == 1 else tree
            print(f"{indent}Class: {class_name}")
            return

        feature_name = feature_names[tree['feature']]
        print(f"{indent}{feature_name} <= {tree['threshold']:.3f} "
              f"(Gain Ratio: {tree['gain_ratio']:.4f})")
        print(f"{indent}  Left -> ", end="")
        self.print_tree(tree['left'], indent + "    ", feature_names)
        print(f"{indent}  Right -> ", end="")
        self.print_tree(tree['right'], indent + "    ", feature_names)


print("\n" + "=" * 60)
print("РУЧНАЯ РЕАЛИЗАЦИЯ C4.5 НА ДАННЫХ BREAST CANCER")
print("=" * 60)

# Обучение ручной модели
manual_c45 = C45Manual(min_samples_split=5, max_depth=4)
manual_c45.fit(X_train, y_train, feature_names=feature_names)

# Печать дерева
print("\nПостроенное дерево (ручная реализация):")
manual_c45.print_tree()

# Предсказание для тестовых данных
predictions_manual = manual_c45.predict(X_test)
accuracy_manual = np.mean(predictions_manual == y_test)
print(f"\nТочность на тестовых данных (ручная реализация): {accuracy_manual:.4f}")

print("\n" + "=" * 60)
print("СРАВНЕНИЕ С BIBLIOTECHNОЙ РЕАЛИЗАЦИЕЙ")
print("=" * 60)

# Использование sklearn (CART, но для сравнения)
from sklearn.tree import DecisionTreeClassifier

# Для sklearn используем criterion='entropy' для имитации поведения ID3/C4.5
sklearn_tree = DecisionTreeClassifier(
    criterion='entropy',
    min_samples_split=5,
    max_depth=4,
    random_state=42
)

sklearn_tree.fit(X_train, y_train)

# Печать дерева sklearn
print("\nДерево (sklearn с entropy):")
tree_rules = export_text(sklearn_tree, feature_names=list(feature_names))
print(tree_rules)

# Предсказания sklearn
predictions_sklearn = sklearn_tree.predict(X_test)
accuracy_sklearn = np.mean(predictions_sklearn == y_test)
print(f"Точность на тестовых данных (sklearn): {accuracy_sklearn:.4f}")

print("\n" + "=" * 60)
print("ДЕТАЛЬНЫЙ АНАЛИЗ")
print("=" * 60)


# Анализ важности признаков
def analyze_feature_importance(X, y, feature_names, top_n=10):
    print(f"Топ-{top_n} признаков по максимальному gain ratio:")

    gain_ratios = []
    manual_c45_temp = C45Manual()

    for feature_idx, feature_name in enumerate(feature_names):
        feature_values = np.unique(X[:, feature_idx])

        if len(feature_values) > 10:
            thresholds = np.random.choice(feature_values[1:], min(10, len(feature_values) - 1), replace=False)
        else:
            thresholds = [(feature_values[i] + feature_values[i + 1]) / 2
                          for i in range(len(feature_values) - 1)]

        max_gain_ratio = 0
        for threshold in thresholds:
            gain_ratio = manual_c45_temp.gain_ratio(X, y, feature_idx, threshold)
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio

        gain_ratios.append((feature_name, max_gain_ratio))

    # Сортируем по убыванию gain ratio
    gain_ratios.sort(key=lambda x: x[1], reverse=True)

    for i, (feature_name, gain_ratio) in enumerate(gain_ratios[:top_n]):
        print(f"{i + 1:2d}. {feature_name:<25}: {gain_ratio:.4f}")


analyze_feature_importance(X_train, y_train, feature_names)

# Сравнение результатов
print(f"\nСравнение результатов:")
print(f"Ручная реализация C4.5:")
print(f"  - Точность: {accuracy_manual:.4f}")
print(f"  - Количество правильных предсказаний: {np.sum(predictions_manual == y_test)}/{len(y_test)}")
print(f"Sklearn Decision Tree:")
print(f"  - Точность: {accuracy_sklearn:.4f}")
print(f"  - Количество правильных предсказаний: {np.sum(predictions_sklearn == y_test)}/{len(y_test)}")

# Анализ ошибок
print(f"\nАнализ ошибок:")
misclassified_manual = np.where(predictions_manual != y_test)[0]
misclassified_sklearn = np.where(predictions_sklearn != y_test)[0]

print(f"Неправильно классифицировано ручной моделью: {len(misclassified_manual)}")
print(f"Неправильно классифицировано sklearn: {len(misclassified_sklearn)}")

# Статистика по классам
print(f"\nСтатистика по классам в тестовой выборке:")
print(f"Класс 0 (Malignant): {np.sum(y_test == 0)} примеров")
print(f"Класс 1 (Benign): {np.sum(y_test == 1)} примеров")

# Матрица ошибок для ручной реализации
from sklearn.metrics import confusion_matrix

print(f"\nМатрица ошибок (ручная реализация):")
cm_manual = confusion_matrix(y_test, predictions_manual)
print(cm_manual)
print("(True Negative, False Positive)")
print("(False Negative, True Positive)")

print(f"\nМатрица ошибок (sklearn):")
cm_sklearn = confusion_matrix(y_test, predictions_sklearn)
print(cm_sklearn)

# ===============================
# ВИЗУАЛИЗАЦИЯ ДЕРЕВЬЕВ
# ===============================

import graphviz
from sklearn.tree import export_graphviz

# 1️⃣ График sklearn дерева
dot_data = export_graphviz(
    sklearn_tree,
    out_file=None,
    feature_names=feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    special_characters=True
)
graph_sklearn = graphviz.Source(dot_data)
graph_sklearn.render("sklearn_tree", format="png", cleanup=True)
graph_sklearn.view()

# 2️⃣ График ручного дерева C4.5
def manual_tree_to_dot(tree, feature_names, class_names, node_id=0, edges=None, labels=None):
    if edges is None:
        edges = []
    if labels is None:
        labels = []

    current_id = node_id

    if isinstance(tree, dict):
        feature_name = feature_names[tree['feature']]
        label = f"{feature_name} <= {tree['threshold']:.3f}\n(Gain Ratio: {tree['gain_ratio']:.4f})"
        labels.append((current_id, label))

        # Левая ветка
        left_id = current_id + 1
        edges.append((current_id, left_id, "True"))
        node_id, edges, labels = manual_tree_to_dot(tree['left'], feature_names, class_names, left_id, edges, labels)

        # Правая ветка
        right_id = node_id + 1000  # уникальные ID для правой ветки
        edges.append((current_id, right_id, "False"))
        node_id, edges, labels = manual_tree_to_dot(tree['right'], feature_names, class_names, right_id, edges, labels)

        return node_id, edges, labels
    else:
        # Лист
        class_name = class_names[tree]
        labels.append((current_id, f"Class: {class_name}"))
        return current_id, edges, labels

# Построение DOT для ручного дерева
node_id, edges, labels = manual_tree_to_dot(manual_c45.tree, feature_names, data.target_names)

dot_str = "digraph C45Manual {\nnode [shape=box, style=filled, color=lightblue];\n"
for node_id, label in labels:
    dot_str += f'  {node_id} [label="{label}"];\n'
for src, dst, edge_label in edges:
    dot_str += f'  {src} -> {dst} [label="{edge_label}"];\n'
dot_str += "}"

manual_graph = graphviz.Source(dot_str)
manual_graph.render("manual_c45_tree", format="png", cleanup=True)
manual_graph.view()
