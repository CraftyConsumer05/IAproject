import cv2
import csv
import pyautogui
import mediapipe as mp
import keyboard
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# myScreenshot = pyautogui.screenshot()
# myScreenshot.save(r'C:\Users\Ef\Desktop\ia\new\screenshot_.png')

def back():
  print("yo")


# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)

    # added
    # cv2.rectangle(image.copy(), (int(100), int(100)), (int(100), int(100)), (255, 12, 145), 2)
    # ends here


    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()

    # added here
    # cv2.rectangle(annotated_image, (100, 150), (500, 600),(0, 255, 0), -1) 
    # ends here

    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=10,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, open('alonsagay.csv', 'w', newline='') as csv_file:
  
  csv_writer = csv.writer(csv_file)
  csv_header = ['landmark','x', 'y', 'z']
  csv_writer.writerow(csv_header)
  while cap.isOpened():
    success, image = cap.read()
    # if keyboard.is_pressed('q'):  # if key 'q' is pressed 
    #   print('You Pressed A Key!')
    #   break
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (80, 150), (280, 350),(0, 255, 0), 2)
    

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        max_x = 0
        min_x = 1
        max_y = 0
        min_y = 1

        if max_x < 0.50 and min_x > 0.10 and max_y < 0.75 and min_y > 0.25:
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
              print('You Pressed q Key!')
              for index, lm in enumerate(mp_hands.HandLandmark):
                lm_point = hand_landmarks.landmark[index]
                lm_row = [lm.name,lm_point.x,lm_point.y,lm_point.z]
                csv_writer.writerow(lm_row)
              myScreenshot = pyautogui.screenshot()
              myScreenshot.save(r'C:\Users\Ef\Desktop\IAFinal\project\screenshot_.png')


        for index, lm in enumerate(mp_hands.HandLandmark):

          for point in hand_landmarks.landmark:
            max_x = max(max_x,point.x)
            min_x = min(min_x,point.x)
            max_y = max(max_y,point.y)
            min_y = min(min_y,point.y)

            
        if max_x < 0.50 and min_x > 0.10 and max_y < 0.75 and min_y > 0.25:

          # Add Bounding Box
          cv2.rectangle(image, (50, 80), (150, 120), (0, 255, 0), -1)

          # Add Button
          cv2.putText(image, 'Capture', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
          cv2.rectangle(image, (80, 150), (280, 350), (0, 0, 255), 2) #change to button


        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    
    

    
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


