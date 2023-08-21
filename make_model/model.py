from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os

# 원본 이미지 파일 경로
image_path = './mini_project/황정민_train/14.jpg'
img = load_img(image_path)
data = img_to_array(img)
samples = expand_dims(data, 0)

datagen = ImageDataGenerator(
    zoom_range=[0.5,1.0],
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    height_shift_range=0.5,
    width_shift_range=0.5)

# 보강된 이미지를 저장할 경로
save_dir = './mini_project/황정민_train/augmented'
os.makedirs(save_dir, exist_ok=True)

# 이미지 보강
it = datagen.flow(samples, batch_size=1)
for i in range(100):
    batch = it.next()
    image = batch[0].astype('uint8')

    # 보강된 이미지 저장
    save_path = os.path.join(save_dir, f'황정민_augmented_{i}.jpg')
    plt.imsave(save_path, image)

# VGG16 모델을 불러와서 이미지 분류 모델을 만들기
from tensorflow.keras.applications import vgg16 as vgg
base_model = vgg.VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(224, 224, 3))

# VGG16 모델의 세 번째 블록에서 마지막 층 추출
last = base_model.get_layer('block3_pool').output

from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras import Model

# 상위 층에 분류층 추가
x = GlobalAveragePooling2D()(last)
x= BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
pred = Dense(30, activation='softmax')(x)
model3 = Model(base_model.input, pred)

for layer in base_model.layers:
     layer.trainable = False
     
model3.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])

history = model3.fit(x=X_train, y=y_train,
                    validation_data=(X_val, y_val),
                    batch_size=batch_size,
                    epochs=20,
                    verbose=1)


# 최적모델 저장
model3.save('team3_new.h5')
