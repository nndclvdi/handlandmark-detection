import cv2
from handDetection import HandDetection

# Buka kamera
cap = cv2.VideoCapture(0)
detector = HandDetection(max_num_hands=2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    allHands = detector.findHandLandMarks(frame, draw=True)
    for i, hand in enumerate(allHands):
        print(f"Tangan {i+1}: {hand}")

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
