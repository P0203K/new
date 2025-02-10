import os
import threading
import time
import numpy as np
import cv2
import mediapipe as mp
from flask import Flask, Response, render_template, request, redirect, url_for, session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import pyttsx3
from threading import Thread

cap = None
cap_lock = threading.Lock()  # Prevent race conditions when accessing cap

def start_camera():
    global cap
    with cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)  # Open the webcam

def stop_camera():
    global cap
    with cap_lock:
        if cap is not None:
            cap.release()  # Release the webcam
            cap = None

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Configure your API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAbhXWje-kecUn6iWxY4X8Nz4gnup3Hj8M"
import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gen_model = genai.GenerativeModel('gemini-1.5-flash')


# Load the machine learning model
# rf_model_dict = pickle.load(open('./model_4.p', 'rb'))
# rf_model = rf_model_dict['model_4']  # Keep this for generate_frames()
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True,
                       min_detection_confidence=0.4, max_num_hands=2)

# Global variables for frame skipping optimization
frame_count = 0
frame_skip = 2  # Process every 2nd frame

def extractData(result) -> list:
    dataLeft = []
    dataRight = []
    totalData = []
    for handType, handLms in zip(result.multi_handedness, result.multi_hand_landmarks):
        if handType.classification[0].label == 'Left':
            for i in range(len(handLms.landmark)):
                x = handLms.landmark[i].x
                y = handLms.landmark[i].y
                dataLeft.append(x)
                dataLeft.append(y)
        else:
            for i in range(len(handLms.landmark)):
                x = handLms.landmark[i].x
                y = handLms.landmark[i].y
                dataRight.append(x)
                dataRight.append(y)

    if len(dataLeft) == 0 and len(dataRight) == 42:
        dataLeft = [0] * 42
    if len(dataRight) == 0 and len(dataLeft) == 42:
        dataRight = [0] * 42
    totalData.extend(dataLeft)
    totalData.extend(dataRight)
    return totalData

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def draw(img, result):
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            img,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame        

def generate_frames():
    global frame_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip processing this frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            draw(frame, result)
            frame_data = extractData(result)
            pred = rf_model.predict([np.asarray(frame_data)])  
            cv2.putText(frame, pred[0], (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            
            # Start a new thread for text-to-speech
            Thread(target=text_to_speech, args=(pred[0],)).start()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
# Initialize Mediapipe components
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4, max_num_hands=2)

# Define actions and initialize model for gestures
actions = np.array(['Hello', 'I', 'Namaste', 'Indian'])
lstm_model = Sequential()  # Renamed from "model" to "lstm_model"
lstm_model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
lstm_model.add(LSTM(128, return_sequences=True, activation='relu'))
lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dense(actions.shape[0], activation='softmax'))
lstm_model.load_weights("action1.h5")

detecting = True

def correct_sentence(words_list):
    prompt = "Make grammatically correct sentence using the given words only: " + " ".join(words_list)
    response = gen_model.generate_content(prompt)
    return response.text

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw(img, result):
    for hand_landmarks in result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                   mp_drawing_styles.get_default_hand_landmarks_style(), 
                                   mp_drawing_styles.get_default_hand_connections_style())

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                               mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                               mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                               mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                               mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                               mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                               mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                               mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                               mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def generate_frames_gesture():
    global cap, detecting  # Reference the global detecting variable
    sequence = []
    sentence = []
    predictions = []
    corrected_sentence = ""
    threshold = 0.5
    last_action_time = time.time()
    stop_threshold = 3.0
    colors = [(0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0)]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame only if detecting is True
            if detecting:
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 
                            last_action_time = time.time()  # Reset timer when an action is detected

                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    corrected_sentence = correct_sentence(sentence)  # Generate corrected sentence
                            else:
                                sentence.append(actions[np.argmax(res)])
                                corrected_sentence = correct_sentence(sentence)
                            
                            corrected_sentence = corrected_sentence.replace('\n', '').replace('\r', '').strip()

                            if len(sentence) > 5: 
                                sentence = sentence[-5:]

                            image = prob_viz(res, actions, image, colors)

            if time.time() - last_action_time > stop_threshold:
                cv2.rectangle(image, (0, 40), (640, 80), (245, 117, 16), -1)
                cv2.putText(image, corrected_sentence, (3, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Prepare image for display
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    print(sentence)
      

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_gesture')
def video_feed_gesture():
    return Response(generate_frames_gesture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection')
def toggle_detection():
    global detecting
    state = request.args.get('state', 'false') == 'true'
    detecting = state
    print(f"Detection state changed to: {detecting}")  # Log the state
    return '', 204  # No content response

@app.route('/translator')
def translator():
    start_camera()
    return render_template('translator.html')

@app.route('/learner')
def learner():
    return render_template('learner.html')

# Route to handle user selection and render flashcards
@app.route('/flashcards', methods=['POST'])
def flashcards():
    choice = request.form.get('choice')
    if choice == 'numbers':
        return render_template('flashno.html')
    elif choice == 'alphabets':
        return render_template('flashAlpha.html')
    else:
        return redirect(url_for('learner')) 

quizzes = {
    'number': {
        'questions': ['/1/1.jpg', '/2/1.jpg', '/3/1.jpg', '/4/1.jpg','/5/2.jpg', '/6/1.jpg', '/7/1.jpg', '/8/1.jpg', '/9/1.jpg',  ],
        'answers': ['1', '2', '3', '4', '5', '6', '7', '8', '9'],  
        'options': [
            ['1', '5', '2', '7'],
            ['3', '2', '4', '1'],
            ['1', '6', '3', '7'],
            ['0', '9', '4', '2'],
            ['5', '6', '3', '8'],
            ['4', '2', '6', '9'],
            ['2', '3', '8', '7'],
            ['5', '3', '8', '4'],
            ['6', '9', '2', '7'],
        ]
    },
    'alphabet': {
        'questions': ['/A/A.jpg', '/B/B.jpg', '/C/C.jpg', '/D/D.jpg', '/E/E.jpg', '/F/F.jpg', '/G/G.jpg', '/H/H.jpg', '/I/I.jpg', '/J/J.jpg', '/K/K.jpg', '/L/L.jpg', '/M/M.jpg', '/N/N.jpg', '/O/O.jpg', '/P/P.jpg', '/Q/Q.jpg', '/R/R.jpg', '/S/S.jpg', '/T/T.jpg', '/U/U.jpg', '/V/V.jpg', '/W/W.jpg', '/X/X.jpg', '/Y/Y.jpg', '/Z/Z.jpg'], 
        'answers': ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'],  
        'options': [
           ['U', 'A', 'R', 'Z'],
['K', 'C', 'B', 'A'],
['T', 'S', 'R', 'C'],
['D', 'M', 'V', 'Q'],
['T', 'Y', 'F', 'E'],
['O', 'S', 'F', 'Y'],
['G', 'O', 'C', 'Y'],
['E', 'C', 'T', 'H'],
['I', 'K', 'F', 'Q'],
['J', 'W', 'O', 'Q'],
['Q', 'K', 'Z', 'I'],
['L', 'C', 'K', 'Z'],
['M', 'C', 'D', 'E'],
['T', 'N', 'S', 'B'],
['O', 'N', 'I', 'F'],
['O', 'V', 'A', 'P'],
['Q', 'Z', 'X', 'V'],
['J', 'C', 'R', 'H'],
['Q', 'S', 'R', 'I'],
['R', 'H', 'T', 'S'],
['E', 'Q', 'U', 'M'],
['P', 'I', 'V', 'Z'],
['S', 'I', 'W', 'J'],
['S', 'Z', 'K', 'X'],
['Y', 'F', 'J', 'N'],
['Z', 'W', 'J', 'N'],

        ]
    }
}

user_answers = []  # Store user's answers for review

@app.route('/quiz')
def render_quiz():
    return render_template('home.html')

@app.route('/sentence')
def render_sentence():
    start_camera()  
    return render_template('sentence.html')

@app.route('/leave_page')
def leave_page():
    stop_camera()  # Stop the camera when leaving the page
    return '', 204  # No content response

@app.route('/quiz/<category>/<int:question_num>', methods=['GET', 'POST'])
def quiz(category, question_num):
    if request.method == 'POST':
        selected_option = request.form['option']
        user_answers.append({'user_answer': selected_option})
        if selected_option == quizzes[category]['answers'][question_num - 1]:
            session['score'] = session.get('score', 0) + 1

        if question_num < len(quizzes[category]['questions']):
            return redirect(url_for('quiz', category=category, question_num=question_num + 1))
        else:
            return redirect(url_for('result', category=category))

    question = quizzes[category]['questions'][question_num - 1]
    options = quizzes[category]['options'][question_num - 1]
    return render_template('quiz.html', category=category, question_num=question_num, question=question, options=options)

@app.route('/result/<category>')
def result(category):
    score = session.get('score', 0)
    total_questions = len(quizzes[category]['questions'])
    correct_percent = score/total_questions*100
    correct_percent =  round(correct_percent, 2)
    print(correct_percent)
    wrong_percent = 100-correct_percent
    print(wrong_percent)
    session.pop('score', None)  
    return render_template('result.html', score=score, total_questions=total_questions, category=category, correct_percent=correct_percent, wrong_percent=wrong_percent)

@app.route('/review/<category>', methods=['GET','POST'])
def review(category):
    print(user_answers)
    review_data = []
    for item in user_answers:
    #     question_num = item['question_num']
        user_answer=item['user_answer']
    #     answers = quizzes[category]['answers']
    #     questions = quizzes[category]['questions']
    #     enumerated_questions = list(enumerate(questions, start=1))
        review_data.append({'user_answer':user_answer})
    print(review_data)
    # return render_template('review.html', review_data=review_data, category=category)

    questions = quizzes[category]['questions']
    answers = quizzes[category]['answers']
    options = quizzes[category]['options']
    enumerated_questions = list(enumerate(questions, start=1))

    combined_data = []
    for index, question in enumerate(enumerated_questions):
        combined_data.append({
            'question': question[1],
            'review_item': review_data[index],
            'correct_answer': answers[index]
    })
    print(combined_data)
    return render_template('review.html', category=category, combined_data=combined_data, answers=answers, options=options, review_data=review_data)



if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    app.run(debug=True)

    cap.release()
    cv2.destroyAllWindows()
