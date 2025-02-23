import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Initialize the face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face(image, face_detector):
    """
    Extract the first detected face from the given image.
    """
    if isinstance(image, str):  # If image path is provided
        image = cv2.imread(image)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]
    return None

def build_transfer_model():
    """
    Build a transfer learning model using MobileNetV2 as the base.
    """
    base_model = MobileNetV2(input_shape=(112, 112, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def predict_image(image):
    """
    Process a single image and predict whether it's Fake or Real.
    """
    model = build_transfer_model()
    
    # Extract face from the image
    face = extract_face(image, face_detector)
    
    if face is not None:
        face = cv2.resize(face, (112, 112))
        face = np.expand_dims(face, axis=0) / 255.0
        prediction = model.predict(face, verbose=0)[0][0]
        
        result = 'Fake' if prediction > 0.5 else 'Real'
        confidence = prediction if result == 'Fake' else 1 - prediction
        return f"{result} (Confidence: {confidence:.2%})"
    else:
        return "No Face Detected in Image"

def predict_video(video_path):
    """
    Process video frame-by-frame and predict whether it's Fake or Real.
    """
    model = build_transfer_model()
    predictions = []
    
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        face = extract_face(frame, face_detector)
        if face is not None:
            face = cv2.resize(face, (112, 112))
            face = np.expand_dims(face, axis=0) / 255.0
            pred = model.predict(face, verbose=0)
            predictions.append(pred[0][0])
        
        processed_frames += 1
    
    video.release()
    
    if predictions:
        avg_prediction = np.mean(predictions)
        result = 'Fake' if avg_prediction > 0.5 else 'Real'
        confidence = avg_prediction if result == 'Fake' else 1 - avg_prediction
        return f"{result} (Confidence: {confidence:.2%})"
    else:
        return "No Face Detected in Video"

def process_media(input_media):
    """
    Process either image or video input and return prediction.
    """
    if input_media is None:
        return "No input provided"
        
    # Check if input is an image or video based on file extension
    if isinstance(input_media, str):
        file_extension = input_media.split('.')[-1].lower()
        if file_extension in ['jpg', 'jpeg', 'png', 'bmp']:
            return predict_image(input_media)
        elif file_extension in ['mp4', 'avi', 'mov', 'wmv']:
            return predict_video(input_media)
        else:
            return "Unsupported file format"
    else:  # If input is a numpy array (direct image)
        return predict_image(input_media)

# Create Gradio interface with tabs for both image and video
with gr.Blocks(title="Deepfake Detection System") as iface:
    gr.Markdown("# Deepfake Detection System")
    gr.Markdown("Upload an image or video to check if it's real or a deepfake.")
    
    with gr.Tabs():
        with gr.Tab("Image Analysis"):
            image_input = gr.Image(type="numpy", label="Upload Image")
            image_button = gr.Button("Analyze Image")
            image_output = gr.Textbox(label="Prediction")
            image_button.click(process_media, inputs=image_input, outputs=image_output)
            
        with gr.Tab("Video Analysis"):
            video_input = gr.Video(label="Upload Video")
            video_button = gr.Button("Analyze Video")
            video_output = gr.Textbox(label="Prediction")
            video_button.click(process_media, inputs=video_input, outputs=video_output)

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)