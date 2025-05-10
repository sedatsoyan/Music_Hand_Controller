
#AUTHOR : SEDAT SOYAN

from random import random

import pygame
import mediapipe as mp
import cv2
import time

from jax.experimental.pallas.ops.tpu.splash_attention import make_random_mask
from numpy.core.defchararray import center
from scipy.io import wavfile
import numpy as np
import math
import random

#dalga fonksiyonu
def draw_sine_wave(img, pt1, pt2, amplitude=20, frequency=0.05, points=115):
    x1, y1 = pt1
    x2, y2 = pt2

    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)

    x_vals = np.linspace(0, distance, points)
    wave_points = []

    for x in x_vals:
        base = math.sin(frequency*x)
        harmonic1 = math.sin(3*frequency*x+math.pi / 4) * 0.5
        harmonic2  = math.sin(5 * frequency * x + math.pi / 2) * 0.3
        noise = random.uniform(-0.8,0.8)

        y = amplitude * (base + harmonic1 +harmonic2+ noise)

        # Rotasyon
        xr = x * math.cos(angle) - y * math.sin(angle)
        yr = x * math.sin(angle) + y * math.cos(angle)
        wave_points.append((int(x1 + xr), int(y1 + yr)))
    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    for i in range(len(wave_points)-1):
        cv2.line(img, wave_points[i], wave_points[i+1], color, 2)

time.sleep(1)
rate,data = wavfile.read(r"C:\Users\sedat\PycharmProjects\datasets\music\KREZUS_-Surreal_dvd-Skins.wav")
mono_data = data if len(data.shape) == 1  else data[:,0]

#music list
music_list = [r"C:\Users\sedat\PycharmProjects\datasets\music\Adore-Did-I-tell-u-that-I-miss-u-_Slowed_-Anime-MV-wlyrics.wav",r"C:\Users\sedat\PycharmProjects\datasets\music\KREZUS_-Surreal_dvd-Skins.wav",r"C:\Users\sedat\PycharmProjects\datasets\music\The-Unknowing_1.wav",r"C:\Users\sedat\PycharmProjects\datasets\music\Şirinamin-were-bamin (1).wav"]
current_track = 0
music_changed_recently = False
last_change_time = 0
pygame.mixer.init()
pygame.mixer.music.load(music_list[current_track])


cap = cv2.VideoCapture(0)
cv2.namedWindow("mirror",cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("mirror",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.5,min_tracking_confidence=0.5)

#mp_drawing_spec = mp_drawing.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=2)

music_started = False
while True:


    color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    color1 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
    mp_drawing_Spec2 = mp_drawing.DrawingSpec(color, thickness=2, circle_radius=2)
    mp_drawing_spec = mp_drawing.DrawingSpec(color1, thickness=2, circle_radius=2)
    ret, frame = cap.read()
    # AYNALAMA
    flipped_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    #yüz çizimi
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(flipped_frame,face_landmarks,mp_face_mesh.FACEMESH_TESSELATION,landmark_drawing_spec=mp_drawing.DrawingSpec(color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)),thickness=1,circle_radius=1),connection_drawing_spec=mp_drawing.DrawingSpec(color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)),thickness=1))

    #PARMAKLAR ARASI pip MESAFE KONTROLÜ
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]

        height,weight ,_ = flipped_frame.shape
        def get_pixel_coords(landmark):
            return int(landmark.x*weight),int(landmark.y*height)

        hand1_thumb = get_pixel_coords(hand1.landmark[4])
        hand1_index = get_pixel_coords(hand1.landmark[8])

        hand2_thumb = get_pixel_coords(hand2.landmark[4])
        hand2_index = get_pixel_coords(hand2.landmark[8])
        center1 = ((hand1_thumb[0] + hand1_index[0]) // 2 ,(hand1_thumb[1] + hand1_index[1]) //2)
        center2 = ((hand2_thumb[0] + hand2_index[0]) //2,(hand2_thumb[1] + hand2_index[1]) // 2)

        hands_distance = math.hypot(center1[0] - center2[0],center1[1]-center2[1])



        if hands_distance < 40 and not music_changed_recently:
            if music_started:
                current_track = (current_track + 1) % len(music_list)
            pygame.mixer.music.load(music_list[current_track])
            pygame.mixer.music.play()
            music_started=True
            music_changed_recently = True
            last_change_time = time.time()

    if music_changed_recently and time.time() - last_change_time > 1:
        music_changed_recently = False

    if not ret:
        break

    current_time_ms = pygame.mixer.music.get_pos()
    current_sample = int(current_time_ms * rate / 1000)
    if current_sample < len(mono_data):
        window=2048
        start = max(0,current_sample-window //2)
        end = min(len(mono_data),current_sample +window // 2)
        window_data=mono_data[start:end]
        amplitude = np.sqrt(np.mean(np.square(window_data))) / 32768
        amplitude*=200
    else :
        amplitude = 10

    amplitude += random.uniform(-10.0,10.0)
    amplitude = max(10,amplitude)






    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(flipped_frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,landmark_drawing_spec=mp_drawing_Spec2,connection_drawing_spec=mp_drawing_spec)
    marklist = []
    if results.multi_hand_landmarks is not None and  len(results.multi_hand_landmarks) == 2:

        for hand_landmarks in results.multi_hand_landmarks:

            for id,lm in enumerate(hand_landmarks.landmark):
                if id == 4 or id == 8:
                    height,width,_ = flipped_frame.shape
                    x_px =int(lm.x * width)
                    y_px = int(lm.y * height)
                    marklist.append(x_px)
                    marklist.append(y_px)
    dynamic_freq = 0.05 + (amplitude / 10)
    if len(marklist) == 8:
        pt1 = (marklist[0],marklist[1])
        pt2 = (marklist[4],marklist[5])
        draw_sine_wave(flipped_frame,(marklist[0],marklist[1]),(marklist[4],marklist[5]),frequency=dynamic_freq,amplitude=amplitude)

        distance = math.hypot(pt2[0] - pt1[0] , pt2[1]-pt1[1])

        volume = min(1.0,max(0.0,distance / 400))
        pygame.mixer.music.set_volume(volume)
    cv2.imshow("mirror", flipped_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
