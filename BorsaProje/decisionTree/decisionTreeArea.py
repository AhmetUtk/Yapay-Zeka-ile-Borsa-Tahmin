import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Karar Ağacı Regresyon modelini oluştur
decision_tree_model = DecisionTreeRegressor(random_state=42)

# Modeli eğit
decision_tree_model.fit(X_train, y_train)

# Modeli kullanarak tahminler yap
y_pred = decision_tree_model.predict(X_test)

# MSE ve R^2 skorlarını hesapla
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")

# Gerçek ve tahmin değerleri arasındaki farkları vurgulamak için alan doldurma grafiği kullan
plt.figure(figsize=(12, 6))
plt.fill_between(range(len(y_test)), y_test - y_pred, color='lightcoral', alpha=0.5, label='Hata')
plt.xlabel('Örnekler')
plt.ylabel('Hata Değeri')
plt.title('Karar Ağacı Regresyon - Hata Değeri (Alan Doldurma Grafiği)')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("areaPlot.png")
plt.show()
