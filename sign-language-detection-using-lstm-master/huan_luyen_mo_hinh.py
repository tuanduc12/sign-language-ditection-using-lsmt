import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Đường dẫn dữ liệu và các tham số
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['null', 'xin chao', 'cam on', 'xin loi', 'hanh phuc', 'tuyet voi', 'yeu thuong', 'ghet', 'biet on', 'tam biet'])
no_sequences = 100
sequence_length = 30
label_map = {label: num for num, label in enumerate(actions)}

# Đọc dữ liệu
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Chia tập dữ liệu
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# TensorBoard và EarlyStopping
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Xây dựng mô hình
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(30, 1662)))  # Dùng tanh mặc định
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Biên dịch mô hình với learning rate nhỏ hơn
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Huấn luyện
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, callbacks=[tb_callback, early_stopping])

# Đánh giá
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# Lưu mô hình
model.save('action.h5')