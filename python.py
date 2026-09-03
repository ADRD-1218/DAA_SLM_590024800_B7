"""
Simple webcam object detector.
Draws a green box around the largest moving/foreground object in the frame.

Requirements:
    pip install opencv-python

Run:
    python object_detector.py

Controls:
    Press 'q' to quit.
"""

import cv2

def main():
    cap = cv2.VideoCapture(0)  # 0 = default laptop camera
    if not cap.isOpened():
        print("Could not open camera. Try changing the index (0, 1, 2...).")
        return

    # Background subtractor: learns the static background, flags anything new as "foreground"
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)

    min_area = 2000  # ignore tiny noise blobs; raise/lower this to tune sensitivity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels more natural

        fg_mask = back_sub.apply(frame)
        # Clean up the mask a bit
        fg_mask = cv2.medianBlur(fg_mask, 5)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > min_area:
                x, y, w, h = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Object Detector (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()