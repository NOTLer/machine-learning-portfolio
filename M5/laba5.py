import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree as sklearn_plot_tree

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============= РУЧНАЯ РЕАЛИЗАЦИЯ M5 =============

class Node:
    """Узел дерева M5"""

    def __init__(self, depth=0, max_depth=10):
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = False
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.model = None  # Линейная модель для листа
        self.prediction = None  # Среднее значение для листа
        self.n_samples = 0


class M5Tree:
    """Ручная реализация M5 Model Tree"""

    def __init__(self, max_depth=10, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_names = None

    def _calculate_std_reduction(self, y_parent, y_left, y_right):
        """Вычисление уменьшения стандартного отклонения (SDR)"""
        n_parent = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        std_parent = np.std(y_parent)
        std_left = np.std(y_left)
        std_right = np.std(y_right)

        # SDR = std(parent) - (n_left/n_parent * std(left) + n_right/n_parent * std(right))
        sdr = std_parent - (n_left / n_parent * std_left + n_right / n_parent * std_right)
        return sdr

    def _find_best_split(self, X, y):
        """Поиск наилучшего разбиения по всем признакам"""
        best_sdr = -np.inf
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            # Берем только часть порогов для ускорения
            if len(thresholds) > 20:
                thresholds = np.percentile(feature_values, np.linspace(10, 90, 20))

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or \
                        np.sum(right_mask) < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                sdr = self._calculate_std_reduction(y, y_left, y_right)

                if sdr > best_sdr:
                    best_sdr = sdr
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_sdr

    def _build_tree(self, X, y, depth=0):
        """Рекурсивное построение дерева"""
        node = Node(depth=depth, max_depth=self.max_depth)
        node.n_samples = len(y)

        # Условия остановки
        if depth >= self.max_depth or \
                len(y) < self.min_samples_split or \
                np.std(y) < 1e-7:
            node.is_leaf = True
            node.prediction = np.mean(y)
            # Строим линейную модель для листа
            node.model = self._build_linear_model(X, y)
            return node

        # Поиск наилучшего разбиения
        feature_idx, threshold, sdr = self._find_best_split(X, y)

        if feature_idx is None or sdr <= 0:
            node.is_leaf = True
            node.prediction = np.mean(y)
            node.model = self._build_linear_model(X, y)
            return node

        # Разбиение
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        node.split_feature = feature_idx
        node.split_value = threshold

        # Рекурсивное построение поддеревьев
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def _build_linear_model(self, X, y):
        """Построение линейной модели для листа"""
        if len(y) < 2:
            return None

        try:
            model = LinearRegression()
            model.fit(X, y)
            return model
        except:
            return None

    def fit(self, X, y, feature_names=None):
        """Обучение дерева"""
        self.feature_names = feature_names
        self.root = self._build_tree(X, y)
        return self

    def _predict_sample(self, x, node):
        """Предсказание для одного примера"""
        if node.is_leaf:
            # Используем линейную модель если есть, иначе среднее
            if node.model is not None:
                try:
                    return node.model.predict(x.reshape(1, -1))[0]
                except:
                    return node.prediction
            return node.prediction

        if x[node.split_feature] <= node.split_value:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """Предсказание для набора примеров"""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _count_nodes(self, node):
        """Подсчет узлов в дереве"""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def get_n_nodes(self):
        """Получение общего количества узлов"""
        return self._count_nodes(self.root)

    def _get_depth(self, node):
        """Получение глубины дерева"""
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))

    def get_depth(self):
        """Получение глубины дерева"""
        return self._get_depth(self.root)


# ============= ВИЗУАЛИЗАЦИЯ ДЕРЕВА =============

def plot_tree(tree, feature_names, title, ax=None, max_depth_show=3):
    """Визуализация структуры дерева M5"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))

    def get_node_count(node, current_depth=0):
        if node is None or current_depth > max_depth_show:
            return 0
        if node.is_leaf:
            return 1
        return 1 + get_node_count(node.left, current_depth + 1) + get_node_count(node.right, current_depth + 1)

    def plot_node(node, x, y, dx, depth=0):
        if node is None or depth > max_depth_show:
            return

        # Цвет в зависимости от типа узла
        if node.is_leaf:
            color = '#90EE90'  # светло-зеленый для листьев
            if node.model is not None:
                label = f'Leaf (LM)\nn={node.n_samples}\nμ={node.prediction:.2f}'
            else:
                label = f'Leaf (Mean)\nn={node.n_samples}\nμ={node.prediction:.2f}'
        else:
            color = '#87CEEB'  # голубой для внутренних узлов
            feat_name = feature_names[node.split_feature] if feature_names else f'X{node.split_feature}'
            label = f'{feat_name} <= {node.split_value:.2f}\nn={node.n_samples}'

        # Рисуем узел
        box = Rectangle((x - dx / 2, y - 0.4), dx, 0.8,
                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, weight='bold')

        if not node.is_leaf and depth < max_depth_show:
            # Рисуем ветви
            y_child = y - 2
            dx_child = dx / 2

            if node.left:
                x_left = x - dx / 2
                ax.plot([x, x_left], [y - 0.4, y_child + 0.4], 'k-', linewidth=2)
                ax.text((x + x_left) / 2 - 0.2, (y + y_child) / 2, 'True',
                        fontsize=8, style='italic')
                plot_node(node.left, x_left, y_child, dx_child, depth + 1)

            if node.right:
                x_right = x + dx / 2
                ax.plot([x, x_right], [y - 0.4, y_child + 0.4], 'k-', linewidth=2)
                ax.text((x + x_right) / 2 + 0.2, (y + y_child) / 2, 'False',
                        fontsize=8, style='italic')
                plot_node(node.right, x_right, y_child, dx_child, depth + 1)

    # Начинаем рисовать с корня
    plot_node(tree.root, 0, 0, 8)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-2 * (max_depth_show + 1), 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14, weight='bold', pad=20)

    return ax


# ============= ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ =============

print("=" * 70)
print("ЗАГРУЗКА РЕАЛЬНОГО ДАТАСЕТА: California Housing")
print("=" * 70)

# Загружаем датасет
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

print(f"\nРазмер датасета: {X.shape[0]} примеров, {X.shape[1]} признаков")
print(f"Признаки: {', '.join(feature_names)}")
print(f"Целевая переменная: Median House Value (в $100,000)")
print(f"Диапазон значений y: [{y.min():.2f}, {y.max():.2f}]")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nОбучающая выборка: {X_train.shape[0]} примеров")
print(f"Тестовая выборка: {X_test.shape[0]} примеров")

# ============= ОБУЧЕНИЕ МОДЕЛЕЙ =============

print("\n" + "=" * 70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 70)

print("\n1. Обучение ручной реализации M5...")
manual_m5 = M5Tree(max_depth=5, min_samples_split=20, min_samples_leaf=10)
manual_m5.fit(X_train, y_train, feature_names)
print(f"   Глубина дерева: {manual_m5.get_depth()}")
print(f"   Количество узлов: {manual_m5.get_n_nodes()}")


# Обычная линейная регрессия для сравнения
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# DecisionTree регрессор (похож на M5, но без линейных моделей в листьях)
dt_model = DecisionTreeRegressor(max_depth=5, min_samples_split=20,
                                 min_samples_leaf=10, random_state=42)
dt_model.fit(X_train, y_train)

print("\nВсе модели обучены!")

# ============= ПРЕДСКАЗАНИЯ =============

print("\n" + "=" * 70)
print("ПРЕДСКАЗАНИЯ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 70)

y_pred_manual = manual_m5.predict(X_test)
y_pred_linear = linear_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)


# ============= МЕТРИКИ =============

def calculate_metrics(y_true, y_pred, model_name):
    """Вычисление метрик качества"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


metrics_manual = calculate_metrics(y_test, y_pred_manual, "Ручная реализация M5")
metrics_linear = calculate_metrics(y_test, y_pred_linear, "Linear Regression")
metrics_dt = calculate_metrics(y_test, y_pred_dt, "Decision Tree")

# ============= ВИЗУАЛИЗАЦИЯ =============

print("\n" + "=" * 70)
print("ПОСТРОЕНИЕ ГРАФИКОВ")
print("=" * 70)

# График 1: Структуры деревьев
fig = plt.figure(figsize=(18, 8))

ax1 = plt.subplot(1, 2, 1)
plot_tree(manual_m5, feature_names, "Ручная реализация M5 Tree", ax1, max_depth_show=3)

ax2 = plt.subplot(1, 2, 2)

sklearn_plot_tree(dt_model, feature_names=feature_names, filled=True,
                  ax=ax2, fontsize=8, max_depth=3)
ax2.set_title("sklearn DecisionTreeRegressor", fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('m5_trees_comparison.png', dpi=300, bbox_inches='tight')
print("✓ График 1 сохранен: m5_trees_comparison.png")
plt.show()

# График 2: Фактические vs Предсказанные значения
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_data = [
    (y_pred_manual, "Ручная M5", metrics_manual),
    (y_pred_linear, "Linear Regression", metrics_linear),
    (y_pred_dt, "Decision Tree", metrics_dt)
]

for idx, (y_pred, title, metrics) in enumerate(models_data):
    axes[idx].scatter(y_test, y_pred, alpha=0.5, s=20)
    axes[idx].plot([y_test.min(), y_test.max()],
                   [y_test.min(), y_test.max()],
                   'r--', lw=2, label='Идеальная линия')
    axes[idx].set_xlabel('Фактические значения', fontsize=11)
    axes[idx].set_ylabel('Предсказанные значения', fontsize=11)
    axes[idx].set_title(f'{title}\nR²={metrics["R2"]:.4f}', fontsize=12, weight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('m5_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# График 3: Распределение ошибок
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (y_pred, title, _) in enumerate(models_data):
    errors = y_test - y_pred
    axes[idx].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('Ошибка предсказания', fontsize=11)
    axes[idx].set_ylabel('Частота', fontsize=11)
    axes[idx].set_title(f'{title}\nСреднее: {np.mean(errors):.4f}',
                        fontsize=12, weight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('m5_errors_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# График 4: Сравнение метрик
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_names = ['RMSE', 'MAE', 'R2']
model_names = ['M5 (ручная)', 'Linear Reg', 'Decision Tree']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, metric in enumerate(metrics_names):
    values = [metrics_manual[metric], metrics_linear[metric], metrics_dt[metric]]
    bars = axes[idx].bar(model_names, values, color=colors)
    axes[idx].set_title(f'Сравнение по {metric}', fontsize=12, weight='bold')
    axes[idx].set_ylabel(metric, fontsize=11)
    axes[idx].grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width() / 2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    axes[idx].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('m5_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# График 5: Важность признаков (для ручной M5)
def get_feature_importance(tree, n_features):
    """Вычисление важности признаков"""
    importance = np.zeros(n_features)

    def traverse(node):
        if node is None or node.is_leaf:
            return
        importance[node.split_feature] += node.n_samples
        traverse(node.left)
        traverse(node.right)

    traverse(tree.root)
    importance = importance / importance.sum()
    return importance


importance_m5 = get_feature_importance(manual_m5, len(feature_names))
importance_dt = dt_model.feature_importances_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# M5 важность
sorted_idx = np.argsort(importance_m5)
ax1.barh(np.array(feature_names)[sorted_idx], importance_m5[sorted_idx], color='#FF6B6B')
ax1.set_xlabel('Важность', fontsize=11)
ax1.set_title('Важность признаков: M5 (ручная)', fontsize=12, weight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# DecisionTree важность
sorted_idx = np.argsort(importance_dt)
ax2.barh(np.array(feature_names)[sorted_idx], importance_dt[sorted_idx], color='#45B7D1')
ax2.set_xlabel('Важность', fontsize=11)
ax2.set_title('Важность признаков: DecisionTree', fontsize=12, weight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('m5_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ График 5 сохранен: m5_feature_importance.png")
plt.show()

# График 6: Residual plot (график остатков)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (y_pred, title, _) in enumerate(models_data):
    residuals = y_test - y_pred
    axes[idx].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('Предсказанные значения', fontsize=11)
    axes[idx].set_ylabel('Остатки', fontsize=11)
    axes[idx].set_title(f'{title}\nResidual Plot', fontsize=12, weight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('m5_residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# ============= ИТОГОВАЯ ТАБЛИЦА =============

print("\n" + "=" * 70)
print("ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
print("=" * 70)

comparison_df = pd.DataFrame({
    'Модель': ['M5 (ручная)', 'Linear Regression', 'Decision Tree'],
    'RMSE': [metrics_manual['RMSE'], metrics_linear['RMSE'], metrics_dt['RMSE']],
    'MAE': [metrics_manual['MAE'], metrics_linear['MAE'], metrics_dt['MAE']],
    'R²': [metrics_manual['R2'], metrics_linear['R2'], metrics_dt['R2']],
    'Количество узлов': [manual_m5.get_n_nodes(), '-', dt_model.tree_.node_count],
    'Глубина дерева': [manual_m5.get_depth(), '-', dt_model.get_depth()]
})

print("\n", comparison_df.to_string(index=False))

