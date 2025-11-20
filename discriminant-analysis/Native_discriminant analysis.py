# Реализация ДА
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ------------------------------
# 1) Загрузка данных breast_cancer
# ------------------------------
data = load_breast_cancer()
X_full = data.data
y_full = data.target

# Возьмем все признаки для лучшего разделения
X = X_full
y = y_full

print(f"Размерность исходных данных: {X.shape}")
print(f"Количество классов: {len(np.unique(y))}")
print(f"Распределение классов: {np.bincount(y)}")
print(f"Названия классов: {data.target_names}\n")

# ------------------------------
# 2) Разделение на подмножества
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Разделяем обучающую выборку по классам
M1 = X_train[y_train == 0]  # класс 0 (злокачественные)
M2 = X_train[y_train == 1]  # класс 1 (доброкачественные)
M0 = X_test                 # тестовая выборка

n1 = M1.shape[0]
n2 = M2.shape[0]
n0 = M0.shape[0]
p = X.shape[1]

print(f"Размерности: n1={n1}, n2={n2}, n0={n0}, p={p}")
print(f"M1 (класс 0 - злокачественные): {n1} samples")
print(f"M2 (класс 1 - доброкачественные): {n2} samples")
print(f"M0 (тестовая выборка): {n0} samples\n")

# ------------------------------
# 3) Средние значения по признакам
# ------------------------------
m1 = np.sum(M1, axis=0) / n1
m2 = np.sum(M2, axis=0) / n2

print("m1 (средние по M1 - класс 0):", np.round(m1[:5], 4), "...")
print("m2 (средние по M2 - класс 1):", np.round(m2[:5], 4), "...\n")

# ------------------------------
# 4) Ковариационные матрицы для каждого множества
# ------------------------------
def covariance_matrix_manual(M, mean_vec):
    n_k = M.shape[0]
    C = M - mean_vec
    S = (C.T @ C) / (n_k - 1)
    return S

S1 = covariance_matrix_manual(M1, m1)
S2 = covariance_matrix_manual(M2, m2)

print("Размер S1:", S1.shape)
print("Размер S2:", S2.shape)

# ------------------------------
# 5) Объединённая ковариационная матрица
# ------------------------------
S_comb = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
print("Размер S_comb:", S_comb.shape)

# ------------------------------
# 6) Обратная матрица S^{-1}
# ------------------------------
det_S = np.linalg.det(S_comb)
print(f"det(S_comb) = {det_S:.6e}")

if abs(det_S) < 1e-12:
    # Если матрица сингулярная, используем псевдообратную
    S_inv = np.linalg.pinv(S_comb)
    print("Использована псевдообратная матрица")
else:
    S_inv = np.linalg.inv(S_comb)
    print("Использована обычная обратная матрица")

# ------------------------------
# 7) Вектор дискриминантных множителей
# ------------------------------
A = S_inv @ (m2 - m1)  # m2 - m1 для правильного направления
print("A (вектор дискриминантных коэффициентов):", np.round(A[:5], 6), "...")

# ------------------------------
# 8) Значения дискриминантной функции
# ------------------------------
F1 = M1 @ A
F2 = M2 @ A
F0 = M0 @ A

print("F1 (для M1 - класс 0):", np.round(F1[:5], 6), "...")
print("F2 (для M2 - класс 1):", np.round(F2[:5], 6), "...")

# ------------------------------
# 9) Средние значения F и константа дискриминации
# ------------------------------
mean_F1 = np.sum(F1) / n1
mean_F2 = np.sum(F2) / n2
F_const = (n1 * mean_F1 + n2 * mean_F2) / (n1 + n2)

print(f"mean(F1) = {mean_F1:.6f}")
print(f"mean(F2) = {mean_F2:.6f}")
print(f"F_const = {F_const:.6f}")

# Проверим разделимость
print(f"\nРазность средних: {mean_F2 - mean_F1:.6f}")
print(f"F1 диапазон: [{np.min(F1):.6f}, {np.max(F1):.6f}]")
print(f"F2 диапазон: [{np.min(F2):.6f}, {np.max(F2):.6f}]")

# ------------------------------
# 10) Классификация вручную
# ------------------------------
def classify_manual(F, F_const):
    # F_i > F_const -> класс 1 (доброкачественные), иначе класс 0 (злокачественные)
    return np.array([1 if f > F_const else 0 for f in F])

FF = np.concatenate([F1, F2, F0])
y_manual_all = classify_manual(FF, F_const)

# Истинные метки в том же порядке
y_true_all = np.concatenate([
    np.zeros(n1),  # класс 0 для M1
    np.ones(n2),   # класс 1 для M2  
    y_test         # истинные метки для теста
])

# ------------------------------
# 11) Библиотечная LDA для сравнения
# ------------------------------
X_stack = np.vstack([M1, M2, M0])
y_stack_train = np.concatenate([np.zeros(n1), np.ones(n2)])

lda = LinearDiscriminantAnalysis()
lda.fit(X_stack[:n1+n2], y_stack_train)
y_pred_lda = lda.predict(X_stack)

# Коэффициенты библиотечной LDA для сравнения
print(f"\nБиблиотечная LDA coefficients: {lda.coef_[0][:5]} ...")
print(f"Библиотечная LDA intercept: {lda.intercept_[0]:.6f}")

# ------------------------------
# 12) Подсчёт ошибок
# ------------------------------
def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred) * 100

# Ошибки на обучающей выборке
train_error_manual = error_rate(y_true_all[:n1+n2], y_manual_all[:n1+n2])
train_error_lda = error_rate(y_true_all[:n1+n2], y_pred_lda[:n1+n2])

# Ошибки на тестовой выборке
test_error_manual = error_rate(y_true_all[n1+n2:], y_manual_all[n1+n2:])
test_error_lda = error_rate(y_true_all[n1+n2:], y_pred_lda[n1+n2:])

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
print("="*60)

print("\n=== Ручная классификация на основе F_const ===")
print(f"Ошибка на обучающей выборке: {train_error_manual:.2f}%")
print(f"Ошибка на тестовой выборке: {test_error_manual:.2f}%")

print("\n=== Библиотечная LDA (sklearn) ===")
print(f"Ошибка на обучающей выборке: {train_error_lda:.2f}%")
print(f"Ошибка на тестовой выборке: {test_error_lda:.2f}%")

# ------------------------------
# 13) Визуализация
# ------------------------------
plt.figure(figsize=(15, 5))

# График 1: дискриминантные scores
plt.subplot(1, 3, 1)
indices_M1 = np.arange(1, n1 + 1)
indices_M2 = np.arange(n1 + 1, n1 + n2 + 1)
indices_M0 = np.arange(n1 + n2 + 1, n1 + n2 + n0 + 1)

plt.scatter(indices_M1, F1, color='red', marker='o', label='M1 (класс 0)', alpha=0.7)
plt.scatter(indices_M2, F2, color='green', marker='s', label='M2 (класс 1)', alpha=0.7)
plt.scatter(indices_M0, F0, color='blue', marker='^', label='M0 (тест)', alpha=0.7)
plt.axhline(F_const, color='black', linestyle='--', label=f'F_const = {F_const:.2f}')
plt.title("Дискриминантные scores по объектам")
plt.xlabel("Номер объекта")
plt.ylabel("F_i")
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: гистограмма распределения
plt.subplot(1, 3, 2)
plt.hist(F1, bins=20, alpha=0.7, color='red', label='Класс 0', density=True)
plt.hist(F2, bins=20, alpha=0.7, color='green', label='Класс 1', density=True)
plt.axvline(F_const, color='black', linestyle='--', label=f'F_const')
plt.xlabel('Дискриминантная функция F')
plt.ylabel('Плотность')
plt.title('Распределение дискриминантных scores')
plt.legend()
plt.grid(True, alpha=0.3)

# График 3: boxplot по классам
plt.subplot(1, 3, 3)
boxplot_data = [F1, F2]
plt.boxplot(boxplot_data, tick_labels=['Класс 0', 'Класс 1'])
plt.ylabel('Дискриминантная функция F')
plt.title('Boxplot по классам')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------
# 14) Детальный анализ нескольких примеров
# ------------------------------
print("\n" + "="*60)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРВЫХ 5 ТЕСТОВЫХ ОБРАЗЦОВ")
print("="*60)

for i in range(min(5, n0)):
    true_class = "злокачественная" if y_test[i] == 0 else "доброкачественная"
    pred_manual = "злокачественная" if y_manual_all[n1+n2+i] == 0 else "доброкачественная"
    pred_lda = "злокачественная" if y_pred_lda[n1+n2+i] == 0 else "доброкачественная"
    
    print(f"Образец {i+1}:")
    print(f"  Истинный класс: {true_class}")
    print(f"  Ручная классификация: {pred_manual} (F={F0[i]:.4f})")
    print(f"  Библиотечная LDA: {pred_lda}")
    print(f"  Верно: {'ДА' if y_manual_all[n1+n2+i] == y_test[i] else 'НЕТ'} (ручная), {'ДА' if y_pred_lda[n1+n2+i] == y_test[i] else 'НЕТ'} (LDA)")
    print()