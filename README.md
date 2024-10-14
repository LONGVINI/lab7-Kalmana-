<h2>Початкові етапи дослідження фільтра Калмана</h2>

<p>Мета цього дослідження полягала у вивченні фільтра Калмана та його застосуванні для обробки шумних даних. Основними завданнями стали:</p>

<ul>
    <li>Знайомство з теорією фільтра Калмана та його застосуванням для фільтрації шуму у вимірюваннях.</li>
    <li>Аналіз параметрів фільтра та їх впливу на точність і стабільність згладжування.</li>
    <li>Дослідження ефектів різних параметрів на дисперсію шуму до та після фільтрації.</li>
</ul>

<h2>Програмний код</h2>

<p>Цей код створює інтерактивний додаток на основі Streamlit для візуалізації роботи Калман-фільтра. Код генерує зашумлений синусоїдальний сигнал і показує, як фільтр обробляє цей сигнал, відображаючи чистий сигнал, шум і оцінки фільтра.</p>

<p><strong>1. Імпорт бібліотек:</strong> використовується <code>NumPy</code> для обчислень, <code>Matplotlib</code> для побудови графіків і <code>Streamlit</code> для створення інтерфейсу веб-додатка.</p>

<p><strong>2. Інтерфейс Streamlit:</strong> код створює бічну панель із слайдерами, що дозволяють користувачу налаштовувати параметри сигналу (частота, амплітуда, зсув, інтервал дискретизації, тривалість) та параметри шуму і фільтра.</p>

<p><strong>3. Налаштування Калман-фільтра:</strong> клас <code>KalmanFilter</code> містить методи <code>predict</code> для прогнозування стану і <code>update</code> для оновлення оцінок на основі нових вимірювань. Це дозволяє фільтру зменшувати вплив шуму на сигнал.</p>

<p><strong>4. Генерація сигналу:</strong> синусоїдальний сигнал генерується з додаванням випадкового шуму. Всі параметри сигналу можна налаштувати через слайдери на панелі Streamlit.</p>

<p><strong>5. Застосування Калман-фільтра:</strong> на основі фільтрації зашумленого сигналу створюються оцінки для кожного кроку часу. Це дозволяє порівняти зашумлений сигнал, реальний сигнал і фільтровані оцінки.</p>

<p><strong>6. Візуалізація результатів:</strong> за допомогою <code>Matplotlib</code> будується графік, який показує три криві: зашумлений сигнал, істинний синусоїдальний сигнал і оцінки Калман-фільтра. Графік вбудовується в інтерфейс Streamlit, що дозволяє користувачу бачити оновлення в реальному часі.</p>

<p>Таким чином, користувач може взаємодіяти з параметрами сигналу та фільтра і спостерігати за змінами в роботі Калман-фільтра на різних наборах даних.</p>

``` python
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# === Streamlit UI Setup ===
st.title("Kalman Filter Visualization")

# Signal parameters
frequency = st.sidebar.slider("Frequency (Hz)", 0.1, 5.0, 1.0)
amplitude = st.sidebar.slider("Amplitude", 0.1, 10.0, 5.0)
offset = st.sidebar.slider("Offset", 0.0, 20.0, 10.0)
sampling_interval = st.sidebar.slider("Sampling Interval (s)", 0.001, 0.1, 0.001)
total_time = st.sidebar.slider("Total Time (s)", 0.1, 5.0, 1.0)

# Noise parameters
noise_variance = st.sidebar.slider("Noise Variance", 1.0, 50.0, 16.0)
noise_std_dev = np.sqrt(noise_variance)

# Filter parameters
Q_value = st.sidebar.slider("Process Noise Covariance (Q)", 0.1, 10.0, 1.0)
R_value = st.sidebar.slider("Measurement Noise Covariance (R)", 0.1, 50.0, 10.0)
P_value = st.sidebar.slider("Initial Estimation Error Covariance (P)", 0.1, 10.0, 1.0)

# Kalman Filter Class
class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.x = x

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

# Generate Signal
time_steps = np.arange(0, total_time, sampling_interval)
true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]

# Apply Kalman Filter
F = np.array([[1]])
H = np.array([[1]])
Q = np.array([[Q_value]])
R = np.array([[R_value]])
P = np.array([[P_value]])
x = np.array([[0]])

kf = KalmanFilter(F, H, Q, R, P, x)
kalman_estimates = []

for measurement in noisy_signal:
    kf.predict()
    estimate = kf.update(measurement)
    kalman_estimates.append(estimate[0][0])

# Plot Results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_steps, noisy_signal, label='Noisy Signal', color='orange', linestyle='-', alpha=0.6)
ax.plot(time_steps, true_signal, label='True Signal (Sine Wave)', linestyle='--', color='blue')
ax.plot(time_steps, kalman_estimates, label='Kalman Filter Estimate', color='green')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Value')
ax.set_title('Kalman Filter Applied to a Noisy Sinusoidal Wave')
ax.legend()
ax.grid()

# Display Plot
st.pyplot(fig)
```

<Виконання дослідження>

<p>У програмі Калман-фільтр має три ключових параметри: <strong>Процесна коваріація шуму (Q)</strong>, <strong>Коваріація шуму вимірювання (R)</strong> та <strong>Початкова коваріація помилки оцінки (P)</strong>. Вони визначають, як фільтр реагує на шум у сигналі та наскільки точно він відслідковує динаміку сигналу. Розглянемо кожен із цих параметрів детальніше:</p>

<h3>Process Noise Covariance (Q)</h3>
<p><strong>Процесна коваріація шуму (Q)</strong> описує рівень довіри до моделі системи, тобто наскільки вона передбачає динаміку сигналу. Високі значення <code>Q</code> означають, що модель допускає велику невизначеність у процесі, вважаючи, що сигнал може значно змінюватися з кожним кроком.</p>
<ul>
  <li><strong>Високе значення Q:</strong> фільтр буде гнучкіше реагувати на зміни в сигналі, що дозволяє йому швидше адаптуватися до різких змін, але також може призвести до більшої нестабільності в оцінках.

      <img src="Screenshots/1.jpg" alt="Запуск Docker контейнера"/>
</p>
<p align="center">
    Високе значення процесної коваріації шуму
</p></li>
  <li><strong>Низьке значення Q:</strong> фільтр припускає, що зміни сигналу більш передбачувані. Це може згладити вихід, але фільтр може не встигати за швидкими змінами сигналу.<p align="center">
  <img src="Screenshots/1.jpg" alt="Запуск Docker контейнера"/>
</p>
<p align="center">
    Високе значення процесної коваріації шуму
</p></li>
</ul>

<p align="center">
  <img src="Screenshots/1.jpg" alt="Запуск Docker контейнера"/>
</p>
<p align="center">
    Високе значення процесної коваріації шуму
</p>

<p align="center">
  <img src="Screenshots/2.jpg" alt="Запуск Docker контейнера"/>
</p>
<p align="center">
    Низька значення процесної коваріації шуму
</p>

<h3>Measurement Noise Covariance (R)</h3>
<p><strong>Коваріація шуму вимірювання (R)</strong> характеризує рівень довіри до вимірюваних даних. Це означає, як фільтр сприймає шум у вимірюваннях. Якщо <code>R</code> має високе значення, фільтр буде вважати, що вимірювання дуже шумні і буде надавати менше значення отриманим даним.</p>
<ul>
  <li><strong>Високе значення R:</strong> фільтр покладається більше на попередні оцінки, аніж на нові вимірювання, що призводить до більш згладжених оцінок.</li>
  <li><strong>Низьке значення R:</strong> фільтр більше довіряє вимірюванням, що дозволяє йому швидше реагувати на нові дані, але робить його більш чутливим до шуму.</li>
</ul>

<h3>Initial Estimation Error Covariance (P)</h3>
<p><strong>Початкова коваріація помилки оцінки (P)</strong> задає початковий рівень невпевненості в прогнозі, на якому ґрунтується фільтр. Вона встановлюється на початку процесу фільтрації і визначає, наскільки сильно фільтр реагує на початкові вимірювання.</p>
<ul>
  <li><strong>Високе значення P:</strong> фільтр вважає, що початкова оцінка є неточною. Це змушує фільтр більше довіряти новим вимірюванням на початкових етапах роботи.</li>
  <li><strong>Низьке значення P:</strong> фільтр вважає, що початкова оцінка дуже точна, тому буде повільніше адаптуватися до змін у вимірюваннях на початку.</li>
</ul>

