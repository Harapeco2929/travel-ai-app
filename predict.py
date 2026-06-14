import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# クラス名
class_names = ["city", "culture", "food", "nature", "resort"]

# モデル読み込み
model = tf.keras.models.load_model("travel_model.keras")

# 画像読み込み
img_path = "test.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 予測
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index]

print("分類結果:", predicted_class)
print("自信度:", round(float(confidence) * 100, 2), "%")
