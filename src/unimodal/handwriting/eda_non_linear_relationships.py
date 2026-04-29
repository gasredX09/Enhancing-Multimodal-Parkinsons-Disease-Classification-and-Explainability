import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import PartialDependenceDisplay
import numpy as np

# --- CONFIG ---
FEATURE_CSV = "/ocean/projects/med260006p/mkhelgi/biomedAI/project/outputs/unimodal_handwriting/in_air_on_tablet_acc_jerk/in_air_on_tablet_features_with_acc_jerk.csv"
LABEL_COL = "label"
MAX_FEATURES = 8  # Number of features to visualize in pairplot/partial dependence

# --- LOAD DATA ---
df = pd.read_csv(FEATURE_CSV)
features = [c for c in df.columns if c not in {"drawing_id", "dataset", "subject_id", "ID", "uci_label", "pahaw_label", "label"}]

# --- PAIRPLOT (non-linear patterns) ---
sns.pairplot(df, vars=features[:MAX_FEATURES], hue=LABEL_COL, plot_kws={'alpha':0.6})
plt.suptitle("Pairplot: Non-linear relationships", y=1.02)
plt.tight_layout()
plt.show()

# --- CORRELATION RATIO (η²) ---
def correlation_ratio(categories, measurements):
    categories = np.array(categories)
    measurements = np.array(measurements)
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.mean(cat_measures) if len(cat_measures) > 0 else 0
    y_total_avg = np.sum(y_avg_array * n_array) / np.sum(n_array)
    numerator = np.sum(n_array * (y_avg_array - y_total_avg) ** 2)
    denominator = np.sum((measurements - y_total_avg) ** 2)
    return numerator / denominator if denominator != 0 else 0

print("\nCorrelation ratio (η²) for each feature vs label:")
for feat in features:
    eta2 = correlation_ratio(df[LABEL_COL], df[feat])
    print(f"{feat:25s}: {eta2:.3f}")

# --- DECISION TREE FEATURE IMPORTANCE (non-linear splits) ---
X = df[features].values
X = StandardScaler().fit_transform(X)
y = df[LABEL_COL].values
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)
importances = clf.feature_importances_
print("\nDecision tree feature importances:")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{feat:25s}: {imp:.3f}")

# --- PARTIAL DEPENDENCE PLOTS (non-linear effects) ---
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(clf, X, features=list(range(min(MAX_FEATURES, len(features)))), feature_names=features, ax=ax)
plt.suptitle("Partial Dependence Plots (Decision Tree)")
plt.tight_layout()
plt.show()

# --- Kernel PCA projection (optional, for visualization) ---
try:
    from sklearn.decomposition import KernelPCA
    kpca = KernelPCA(n_components=2, kernel='rbf', random_state=42)
    X_kpca = kpca.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_kpca[:,0], X_kpca[:,1], c=y, cmap='coolwarm', alpha=0.6)
    plt.title('Kernel PCA (RBF) projection')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar(label='Label')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("scikit-learn >= 0.18 required for KernelPCA. Skipping kernel PCA plot.")
