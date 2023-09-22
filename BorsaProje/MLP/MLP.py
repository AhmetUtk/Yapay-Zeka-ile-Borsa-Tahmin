import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Veri kümesini oku
df = pd.read_excel("eregli.xlsx")

# Gereksiz sütunları kaldır
dropData = ["Tarih", "Min(TL)", "Max(TL)", "AOF(TL)", "Hacim(TL)", "Sermaye(mn TL)", "BIST 100", "PiyasaDeğeri(mn TL)", "PiyasaDeğeri(mn USD)", "HalkaAçık PD(mn TL)", "HalkaAçık PD(mn USD)"]
df = df.drop(dropData, axis=1)

# Bağımsız değişkenleri ve hedef değişkeni ayarla
X = df.drop("Kapanış(TL)", axis=1)
y = df["Kapanış(TL)"]

# Veriyi eğitim ve test kümesi olarak bölmek
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendirme (standartlaştırma)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Yapay Sinir Ağı modelini oluştur
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Modeli eğit
model.fit(X_train, y_train)

# Model performansını test et
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse}")

import matplotlib.pyplot as plt

# Test verileri üzerinde tahminleri yap
y_pred = model.predict(X_test)

# Gerçek ve tahmini değerleri bir araya getir
comparison_df = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})

# Grafik oluştur
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', label='Gerçek vs. Tahmin')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek Değerler vs. Tahmin Edilen Değerler')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("MlAcc.png")
plt.show()

