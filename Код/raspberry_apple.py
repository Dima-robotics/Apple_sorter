# Импорт библиотек
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import RPi.GPIO as GPIO
import time

# Загрузка модели
model = load_model("keras_model_latest.h5", compile=False)
class_names = open("labels_latest.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# поворот сервомотора
def set_angle(servo,pin,angle,t): # 10 90 180
    duty = angle/18 +2
    GPIO.output(pin,True)
    servo.ChangeDutyCycle(duty)
    time.sleep(t)
    GPIO.output(pin,False)
    servo.ChangeDutyCycle(0)

# Объявление пинов
GPIO.setmode(GPIO.BCM)
GPIO.setup(23,GPIO.OUT)
separator = GPIO.PWM(23,50)
separator.start(0)
GPIO.setup(24,GPIO.OUT)
turn = GPIO.PWM(24,50)
turn.start(0)
GPIO.setup(25,GPIO.OUT)
sorter = GPIO.PWM(25,50)
sorter.start(0)
set_angle(sorter,25,90,1)

# Старт камеры
vid = cv2.VideoCapture(0)
ret, frame = vid.read()
crop_frame = frame[140:300,210:380]
frame_resized = cv2.resize(crop_frame, (224, 224))
cv2.imwrite("img_res.png",frame_resized)
normalized_image_array = (frame_resized.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array
prediction = model.predict(data)


np.set_printoptions(suppress=True)
# Основной цикл
while True:
    set_angle(separator,23,70,1.9) # Поворот сепаратора
    time.sleep(0.5)
    set_angle(sorter,25,90,1)
    
    # Поворот яблока, чтобы осмотреть его камерой со всех сторон
    goodc = 0
    badc = 0
    for i in range(10):
        for j in range(4): # Пропуск кадров, чтобы исключить возможность "лага" камеры
            ret, frame = vid.read()
        ret, frame = vid.read()

        # Обработка кадра
        crop_frame = frame[140:300,210:380]
        frame_resized = cv2.resize(crop_frame, (224, 224))
        cv2.imwrite("img_res"+str(i)+".png",frame_resized)
        normalized_image_array = (frame_resized.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        # Предсказание
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Вывод результатов
        print("Class:", class_name[2:], end="")
        if class_name.find("Good")!=-1:
            goodc+=confidence_score
        else:
            badc+=confidence_score
        print("Confidence Score:", confidence_score)
        
        if cv2.waitKey(1) == ord('q'):
            break
        set_angle(turn,24,50,0.16)
        time.sleep(0.5)
    print(goodc,badc)
    
    # Финальное решение по яблоку
    if badc>2:
        set_angle(sorter,25,10,1)
    else:
        set_angle(sorter,25,180,1)
    set_angle(sorter,25,90,1)

vid.release()
cv2.destroyAllWindows()

