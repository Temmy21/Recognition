import streamlit as st
import cv2
import numpy as np

class SimpleFaceRecognition:
    def __init__(self):
        # Load the pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_face(self, image):
        """Detect faces in image and return the largest face"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        
        # Get the largest face
        areas = [w * h for (x, y, w, h) in faces]
        largest_face_idx = np.argmax(areas)
        x, y, w, h = faces[largest_face_idx]
        
        # Extract and preprocess face
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        return face_img, (x, y, w, h)


    def get_face_encoding(self, face_img):
        """Create a simple face encoding using pixel values"""
        return face_img.flatten()
    
def main():
    st.title("📸 Simple Face Recognition App")
    
    # Initialize session state
    if 'known_faces' not in st.session_state:
        st.session_state.known_faces = {}
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = SimpleFaceRecognition()
    if 'clear_faces' not in st.session_state:
        st.session_state.clear_faces = False
    
    # Mode selection
    mode = st.radio("Select Mode:", ["Learn New Face", "Recognize Face"], horizontal=True)
    
    if mode == "Learn New Face":
        st.write("### 👨‍🎓 Teaching Mode")
        person_name = st.text_input("Enter person's name:")
        
        if person_name:
            capture = st.camera_input("Take a picture to learn the face")
            
            if capture:
                # Convert the captured image to opencv format
                file_bytes = np.asarray(bytearray(capture.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Detect and process face
                result = st.session_state.recognizer.detect_face(image)
                
                if result is not None:
                    face_img, (x, y, w, h) = result
                    encoding = st.session_state.recognizer.get_face_encoding(face_img)
                    
                    # Store the face encoding
                    st.session_state.known_faces[person_name] = encoding
                    
                    # Draw rectangle around detected face
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    st.success(f"✅ Successfully learned {person_name}'s face!")
                    st.balloons()
                else:
                    st.error("No face detected! Please try again.")
    
    else:  # Recognize Face mode
        st.write("### 🔍 Recognition Mode")
        if not st.session_state.known_faces:
            st.warning("Please teach some faces first!")
        else:
            capture = st.camera_input("Take a picture to recognize faces")
            
            if capture:
                # Convert the captured image to opencv format
                file_bytes = np.asarray(bytearray(capture.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                # Detect and process face
                result = st.session_state.recognizer.detect_face(image)
                
                if result is not None:
                    face_img, (x, y, w, h) = result
                    test_encoding = st.session_state.recognizer.get_face_encoding(face_img)
                    
                    # Find the best match
                    min_dist = float('inf')
                    best_match = None
                    
                    for name, encoding in st.session_state.known_faces.items():
                        dist = np.linalg.norm(encoding - test_encoding)
                        if dist < min_dist:
                            min_dist = dist
                            best_match = name
                    
                    # Draw rectangle and name
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    confidence = max(0, min(100, 100 - (min_dist / 1000)))
                    
                    if confidence > 60:
                        st.success(f"Recognized: {best_match} (Confidence: {confidence:.1f}%)")
                        if confidence > 80:
                            st.write("🌟 High confidence match!")
                        else:
                            st.write("✨ Good match!")
                    else:
                        st.info("Unknown person detected 🤔")
                    
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    st.error("No face detected! Please try again.")
    
    # Display known faces
    if st.session_state.known_faces:
        st.write("### 📋 Known Faces")
        for name in st.session_state.known_faces.keys():
            st.write(f"- {name}")
        
        if st.button("Clear All Known Faces"):
            st.session_state.known_faces = {}
            st.rerun()  # Using the new rerun() method instead of experimental_rerun()

if __name__ == "__main__":
    main()