from PIL import Image
import facenet_pytorch
import numpy as np

class FaceDetector():
    def __init__(self):
        self.mtcnn = facenet_pytorch.MTCNN()
        
    def detect_face(self, image, resized=False, size=96, margin=(0, 0, 0, 0)):
        faces_data = []
        boxes, _ = self.mtcnn.detect(image)

        # Check if no faces are detected
        if boxes is None:
            return faces_data

        for box in boxes:
            box = np.clip(box, 0, min(image.height, image.width))
            left, top, right, bottom = box
            # Add margin to the face region
            left_margin = max(left - margin[0], 0)
            right_margin = min(right + margin[1], image.width)
            top_margin = max(top - margin[2], 0)
            bottom_margin = min(bottom + margin[3], image.height)
            face_image = image.crop((left_margin, top_margin, right_margin, bottom_margin))
            
            if resized:
                element_data = dict()
                element_data["image"] = face_image.resize((size, size))
                element_data["width"] = int(right_margin - left_margin)
                element_data["height"] = int(bottom_margin - top_margin)
                
                faces_data.append(element_data)  # Append inside the loop
                
        return faces_data
