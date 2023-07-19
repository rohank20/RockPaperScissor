import random
import tensorflow as tf
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('saved_models/rps_exp_1', compile=False)
while True:
    actions = ['Rock', 'Paper', 'Scissor']
    gen_num = random.randint(0, 2)
    random_action = actions[gen_num]
    # Read frame from camera
    ret, image_np = cap.read()
    image_np_resized = cv2.resize(image_np, (200, 200))
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np_resized, axis=0)
    result = np.argmax(model.predict(image_np_expanded))
    result_action = actions[result]
    if result - gen_num == 0:
        winner = "It's a tie"
    elif result - gen_num >= 1:
        winner = "You won!"
    else:
        winner = 'Computer won, try again'
    output = cv2.putText(image_np, f'Computer: {random_action} User: {result_action} {winner}', org=(10, 20),
                         fontFace=0, fontScale=0.5, color=(0, 0, 255))
    # Display output
    cv2.imshow('Result', output)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
