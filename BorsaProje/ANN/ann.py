import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
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

# Yapay Sinir Ağı modelini oluştur
model = Sequential()
model.add(Dense(units=256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=64,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=32,activation="relu"))
model.add(Dense(units=16,activation="relu"))
model.add(Dense(units=8,activation="relu"))
model.add(Dense(units=1))

# Modeli derleme
model.compile(loss='mean_squared_error', optimizer='adam')

# Modeli eğit
model.fit(X_train, y_train, epochs=128, batch_size=32, validation_data=(X_test, y_test))

# Model performansını test et
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse}")
print(f"R^2 Skoru: {r2}")

# Gerçek ve tahmin değerlerini aynı çizgi grafiğinde göster
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', marker='o', label='Gerçek vs. Tahmin')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek Değerler vs. Tahmin Edilen Değerler')
plt.legend(loc='best')

# 45 derece doğru eklemek için
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value], c='red', linestyle='--', label='45 Derece Doğru')
plt.legend(loc='best')
plt.grid(True)
plt.savefig("acc.png")
plt.show()

