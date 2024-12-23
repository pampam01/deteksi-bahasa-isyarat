import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)



offset = 20
imgSize = 300

# Folder data utama
base_folder = "data-v2"
if not os.path.exists(base_folder):
    os.makedirs(base_folder)



# Huruf awal untuk folder
current_char = 'J'
folder = os.path.join(base_folder, current_char)
os.makedirs(folder, exist_ok=True)
counter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Handle cropping out-of-bounds
        if y - offset < 0 or x - offset < 0 or y + h + offset > img.shape[0] or x + w + offset > img.shape[1]:
            continue

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord('s'):
        counter += 1
        filename = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filename, imgWhite)
        print(f"Saved: {filename}")

    # Cek apakah sudah mencapai 50 gambar untuk folder saat ini
    if counter >= 50:
        print(f"Folder {current_char} sudah penuh dengan 50 gambar.")

        # Pindah ke folder berikutnya
        current_char = chr(ord(current_char) + 1)
        if current_char > 'Y':  # Jika sudah melewati Y, berhenti
            print("Dataset lengkap dari A sampai Y sudah dibuat.")
            break

        counter = 0
        folder = os.path.join(base_folder, current_char)
        os.makedirs(folder, exist_ok=True)
        print(f"Beralih ke folder: {current_char}")

cap.release()
cv2.destroyAllWindows()



