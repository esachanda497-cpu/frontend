import cv2


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not access webcam.")

        break

    cv2.imshow(
        "SafeStep Webcam Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()
