import cv2
import mediapipe as mp 
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time 
from mediapipe.framework.formats import landmark_pb2
import chromadb
import uuid
from PIL import Image, ImageDraw, ImageFont
import os 


MARGIN = 10  # pixels
FONT_SIZE = 10
FONT_THICKNESS = 4
HANDEDNESS_TEXT_COLOR = (222,92,48) 

class HandSigns():
    def __init__(self):
        self.result = vision.HandLandmarkerResult
        self.live_landmarker = vision.HandLandmarker
        self.dbclient = chromadb.PersistentClient()
        self.collection = self.dbclient.get_or_create_collection(name="my_collection")
        self.create_live_landmarker()
        self.testing = False

    def create_live_landmarker(self):
        def update_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            self.result = result

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model', 'hand_landmarker.task')

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=update_result
        )
        self.live_landmarker = self.live_landmarker.create_from_options(options)

    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.live_landmarker.detect_async(mp_image, int(time.time() * 1000))

    def reset_db(self):
        self.dbclient.delete_collection(name="my_collection")
        self.collection = self.dbclient.get_or_create_collection(name="my_collection")

    def store_to_db(self, label):
        for i, landmark in enumerate(self.result.hand_landmarks):
            category = self.result.handedness[i][0].display_name
            vector = self.coordinates_to_vector(landmark)

            self.collection.add(
                embeddings = [vector],
                metadatas = [{"sign": label, "hand": category}],
                ids = [str(uuid.uuid4())]
            )

    @staticmethod
    def coordinates_to_vector(coords):
        # using the palm base as the reference
        reference_point = np.array([coords[0].x, coords[0].y, coords[0].z]) 

        x_coords = np.array([coord.x for coord in coords]) - reference_point[0]
        y_coords = np.array([coord.y for coord in coords]) - reference_point[1]
        z_coords = np.array([coord.z for coord in coords]) - reference_point[2]

        return np.concatenate([x_coords, y_coords, z_coords])

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        try:
            if detection_result.hand_landmarks == []:
                return rgb_image
            else:
                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness
                annotated_image = np.copy(rgb_image)

                # Loop through the detected hands to visualize.
                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]

                    # Draw the hand landmarks.
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                        for landmark in hand_landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        hand_landmarks_proto,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

                    if self.testing:
                        handedness = handedness_list[idx]
                        vector = self.coordinates_to_vector(hand_landmarks)

                        # Get the top left corner of the detected hand's bounding box.
                        height, width, _ = annotated_image.shape
                        x_coordinates = [landmark.x for landmark in hand_landmarks]
                        y_coordinates = [landmark.y for landmark in hand_landmarks]
                        text_x = int(min(x_coordinates) * width)
                        text_y = int(min(y_coordinates) * height) - 10  # MARGIN is assumed as 10

                        results = self.collection.query(
                            query_embeddings=vector,
                            n_results=1
                        )

                        if results and results['distances'][0][0] < 0.09 and handedness[0].category_name == results['metadatas'][0][0]['hand']:
                            sign = results['metadatas'][0][0]['sign']
                            pil_image = Image.fromarray(annotated_image)
                            draw = ImageDraw.Draw(pil_image)
                            font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/AppleGothic.ttf", 256)
                            draw.text((text_x, text_y), sign, font=font, fill=(255, 255, 255))
                            annotated_image = np.array(pil_image)

                return annotated_image
        except:
            return rgb_image
        
    def close(self):
        self.live_landmarker.close()


if __name__ == '__main__':
    dbclient = chromadb.PersistentClient()
    collection = dbclient.get_or_create_collection(name="my_collection")

    print(collection.peek())