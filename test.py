import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)



detector = HandDetector(maxHands=2)


classifier = Classifier(
    r"C:\Users\Denisa\Desktop\Model\keras_model.h5",
    r"C:\Users\Denisa\Desktop\Model\labels.txt"
)




offset = 20
imgSize = 300



labels = [
    
   
    "Bom",
    "Dia",
    "Prazer",
    "te",
    "conhecer!",
    "My",
    "name",
    "is",
    "D"
    "e"
    "n"
    "i"
    "s"
    "a",
    "I am",
    "hungry!"
    


]



sentence = []

last_prediction = ""
prediction_start_time = 0

# Gesture must remain stable for this amount of time
STABLE_TIME = 1.0

# Prevent repeated gesture detection
cooldown = 1.0
last_added_time = 0

# ============================================================
# TEXT TO SPEECH
# ============================================================

engine = pyttsx3.init()

engine.setProperty("rate", 150)


def speak_sentence(text):

    engine.say(text)
    engine.runAndWait()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, img = cap.read()

    if not success:
        print("Camera could not be accessed.")
        break

    imgOutput = img.copy()

    # --------------------------------------------------------
    # FIND HAND
    # --------------------------------------------------------

    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]

        x, y, w, h = hand["bbox"]

        # ----------------------------------------------------
        # CREATE WHITE IMAGE
        # ----------------------------------------------------

        imgWhite = np.ones(
            (imgSize, imgSize, 3),
            np.uint8
        ) * 255

        # ----------------------------------------------------
        # SAFE CROP
        # ----------------------------------------------------

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)

        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        # ----------------------------------------------------
        # ASPECT RATIO
        # ----------------------------------------------------

        aspectRatio = h / w

        if aspectRatio > 1:

            k = imgSize / h

            wCal = math.ceil(k * w)

            imgResize = cv2.resize(
                imgCrop,
                (wCal, imgSize)
            )

            wGap = math.ceil(
                (imgSize - wCal) / 2
            )

            imgWhite[
                :,
                wGap:wGap + wCal
            ] = imgResize

        else:

            k = imgSize / w

            hCal = math.ceil(k * h)

            imgResize = cv2.resize(
                imgCrop,
                (imgSize, hCal)
            )

            hGap = math.ceil(
                (imgSize - hCal) / 2
            )

            imgWhite[
                hGap:hGap + hCal,
                :
            ] = imgResize

        # ----------------------------------------------------
        # PREDICT GESTURE
        # ----------------------------------------------------

        prediction, index = classifier.getPrediction(
            imgWhite,
            draw=False
        )

        # ----------------------------------------------------
        # CHECK VALID INDEX
        # ----------------------------------------------------

        if index >= 0 and index < len(labels):

            current_prediction = labels[index]

            # ------------------------------------------------
            # STABLE GESTURE DETECTION
            # ------------------------------------------------

            if current_prediction != last_prediction:

                last_prediction = current_prediction
                prediction_start_time = time.time()

            else:

                elapsed = time.time() - prediction_start_time

                # Gesture stayed stable
                if (
                    elapsed >= STABLE_TIME
                    and
                    time.time() - last_added_time > cooldown
                ):

                    # Don't immediately add duplicates
                    if len(sentence) == 0 or sentence[-1] != current_prediction:

                        sentence.append(current_prediction)

                        last_added_time = time.time()

                        print(
                            "Gesture:",
                            current_prediction
                        )

                        print(
                            "Sentence:",
                            " ".join(sentence)
                        )

                        # Reset timer
                        prediction_start_time = time.time()

            # ------------------------------------------------
            # SHOW CURRENT GESTURE
            # ------------------------------------------------

            cv2.rectangle(
                imgOutput,
                (x - offset, y - offset - 70),
                (x - offset + 400, y - offset),
                (0, 255, 0),
                cv2.FILLED
            )

            cv2.putText(
                imgOutput,
                current_prediction,
                (x, y - 20),
                cv2.FONT_HERSHEY_COMPLEX,
                1.5,
                (0, 0, 0),
                2
            )

        # ----------------------------------------------------
        # DRAW HAND BOX
        # ----------------------------------------------------

        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (0, 255, 0),
            4
        )

        # ----------------------------------------------------
        # SHOW CROPPED HAND
        # ----------------------------------------------------

        cv2.imshow(
            "ImageCrop",
            imgCrop
        )

        cv2.imshow(
            "ImageWhite",
            imgWhite
        )

    # ========================================================
    # DISPLAY SENTENCE
    # ========================================================

    sentence_text = " ".join(sentence)

    cv2.rectangle(
        imgOutput,
        (0, 0),
        (imgOutput.shape[1], 80),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        imgOutput,
        sentence_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2
    )

    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        imgOutput,
        "S = Speak | C = Clear | B = Backspace | Q = Quit",
        (20, imgOutput.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    # ========================================================
    # SHOW MAIN WINDOW
    # ========================================================

    cv2.imshow(
        "Sign Language Recognition",
        imgOutput
    )

    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # SPEAK SENTENCE
    # --------------------------------------------------------

    if key == ord("s"):

        if len(sentence) > 0:

            text = " ".join(sentence)

            print("Speaking:", text)

            speak_sentence(text)

    # --------------------------------------------------------
    # CLEAR SENTENCE
    # --------------------------------------------------------

    elif key == ord("c"):

        sentence = []

        print("Sentence cleared.")

    # --------------------------------------------------------
    # BACKSPACE
    # --------------------------------------------------------

    elif key == ord("b"):

        if len(sentence) > 0:

            removed = sentence.pop()

            print(
                "Removed:",
                removed
            )

    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        break


# ============================================================
# RELEASE
# ============================================================

cap.release()

cv2.destroyAllWindows()