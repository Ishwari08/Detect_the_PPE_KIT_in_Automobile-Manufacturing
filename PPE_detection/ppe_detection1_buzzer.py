from ultralytics import YOLO
import cv2
import cvzone
import math
import pygame  # Import pygame for sound

# Initialize pygame mixer for sound
pygame.mixer.init()
buzzer_sound = pygame.mixer.Sound("buzzer.mp3")  # Replace with your buzzer sound file path

# Load the video
cap = cv2.VideoCapture("D:/C DRIVE/Downloads/Project/ppe_detection/videos/video1.mp4")

# Get the width and height of the frames in the video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID'
output_path = 'C:\\Users\\Dell\\Downloads\\output.mp4'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Load the YOLO model
model = YOLO("D:/C DRIVE/Downloads/Project/ppe_detection/best1.pt")

# Class names and corresponding colors
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask',
              'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus',
              'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
              'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

# Define specific colors for each detection
classColors = {
    'Person': (255, 0, 0),           # Blue
    'Hardhat': (238, 130, 238),      # Violet
    'Safety Vest': (0, 255, 0),      # Green
    'Mask': (139, 69, 19),           # Brown
    'Gloves': (0, 255, 255),         # Yellow
    'NO-Hardhat': (0, 0, 255),       # Red
    'NO-Mask': (0, 0, 255),          # Red
    'NO-Safety Vest': (0, 0, 255),   # Red
    'NO-Person': (0, 0, 255),        # Red
    'NO-Gloves': (0, 0, 255)         # Red
}

while True:
    success, img = cap.read()
    if not success:
        break  # Break the loop if no frame is returned

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            # Determine the color based on the detected class
            myColor = classColors.get(currentClass, (255, 255, 255))  # Default to white if not in classColors

            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask', 'NO-Person', 'NO-Gloves']:
                    if not pygame.mixer.get_busy():  # Play buzzer if not already playing
                        buzzer_sound.play()

                # Display bounding box and label
                cvzone.putTextRect(img, f'{currentClass} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    # Write the frame to the output video
    out.write(img)

    # Display the frame
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
