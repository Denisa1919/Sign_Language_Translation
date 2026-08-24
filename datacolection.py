import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

# Detect up to 2 hands
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
counter = 0

# Dataset folder
folder = r"C:\Users\Denisa\Downloads\Sign_language_detection\Data\I am"

os.makedirs(folder, exist_ok=True)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, img = cap.read()

    if not success:
        print("Could not access webcam.")
        break

    imgOutput = img.copy()

    # Detect one or two hands
    hands, img = detector.findHands(img)

    # ========================================================
    # IF HAND(S) DETECTED
    # ========================================================

    if hands:

        # ----------------------------------------------------
        # ONE HAND
        # ----------------------------------------------------

        if len(hands) == 1:

            hand = hands[0]

            x, y, w, h = hand["bbox"]

            # Create 300x300 white image
            imgWhite = np.ones(
                (imgSize, imgSize, 3),
                np.uint8
            ) * 255

            # Safe crop coordinates
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)

            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:

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

                cv2.imshow(
                    "ImageCrop",
                    imgCrop
                )

                cv2.imshow(
                    "ImageWhite",
                    imgWhite
                )

        # ----------------------------------------------------
        # TWO HANDS
        # ----------------------------------------------------

        elif len(hands) == 2:

            hand1 = hands[0]
            hand2 = hands[1]

            x1, y1, w1, h1 = hand1["bbox"]
            x2, y2, w2, h2 = hand2["bbox"]

            # Find the combined bounding box
            xMin = max(0, min(x1, x2) - offset)
            yMin = max(0, min(y1, y2) - offset)

            xMax = min(
                img.shape[1],
                max(x1 + w1, x2 + w2) + offset
            )

            yMax = min(
                img.shape[0],
                max(y1 + h1, y2 + h2) + offset
            )

            # Crop both hands together
            imgCrop = img[
                yMin:yMax,
                xMin:xMax
            ]

            if imgCrop.size != 0:

                cropH, cropW = imgCrop.shape[:2]

                imgWhite = np.ones(
                    (imgSize, imgSize, 3),
                    np.uint8
                ) * 255

                aspectRatio = cropH / cropW

                if aspectRatio > 1:

                    k = imgSize / cropH

                    wCal = math.ceil(
                        k * cropW
                    )

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

                    k = imgSize / cropW

                    hCal = math.ceil(
                        k * cropH
                    )

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

                cv2.imshow(
                    "ImageCrop",
                    imgCrop
                )

                cv2.imshow(
                    "ImageWhite",
                    imgWhite
                )

        # ====================================================
        # DISPLAY NUMBER OF HANDS
        # ====================================================

        cv2.putText(
            imgOutput,
            f"Hands detected: {len(hands)}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            imgOutput,
            "No hands detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ========================================================
    # SHOW CAMERA
    # ========================================================

    cv2.imshow(
        "Image",
        imgOutput
    )

    # ========================================================
    # SAVE IMAGE
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):

        if hands:

            counter += 1

            filename = os.path.join(
                folder,
                f"Image_{int(time.time() * 1000)}.jpg"
            )

            cv2.imwrite(
                filename,
                imgWhite
            )

            print(
                f"Image saved: {filename}"
            )

            print(
                f"Total images: {counter}"
            )

        else:

            print(
                "No hand detected. Image not saved."
            )

    # ========================================================
    # QUIT
    # ========================================================

    if key == ord("q"):
        break


# ============================================================
# RELEASE
# ============================================================

cap.release()
cv2.destroyAllWindows()