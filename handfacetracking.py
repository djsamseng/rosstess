import os
import struct

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import KDTree

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


def load_kdtree():
  dirname = "./faces"
  data = None
  labels = None
  tracking_faces: list[TrackedFace] = []
  for filename in os.listdir(dirname):
    path = os.path.join(dirname, filename)
    if path.endswith(".npy"):
      structures = np.load(path)
      structures = structures.reshape((structures.shape[0], -1))
      name = filename.strip(".npy")
      new_names = np.array([name for _ in range(structures.shape[0])])
      if data is None:
        data = structures
        labels = new_names
      else:
        data = np.concatenate((data, structures))
        labels = np.concatenate((labels, new_names))
      tracking_faces.append(TrackedFace(name=name, face_structures=structures))

  tree = KDTree(data)
  return tree, labels, tracking_faces


class TrackedFace():
  def __init__(self, name:str, face_structures:np.ndarray) -> None:
    self.name = name
    self.face_structures = face_structures
    self.last_face_structure = face_structures[-1]

  def add_face_structure(self, face_structure:np.ndarray, save:bool) -> None:
    self.last_face_structure = face_structure
    if save or self.face_structures.shape[0] < 500:
      print("Adding:", self.name, self.face_structures.shape[0])
      self.face_structures = np.concatenate((self.face_structures, [face_structure]))

  def save(self) -> None:
    np.save("./faces/{0}.npy".format(self.name), self.face_structures)


def draw_face(image, face, show:bool):
  mp_drawing.draw_landmarks(
    image=image,
    landmark_list=face,
    connections=mp_face_mesh.FACEMESH_TESSELATION,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles
    .get_default_face_mesh_tesselation_style())
  mp_drawing.draw_landmarks(
    image=image,
    landmark_list=face,
    connections=mp_face_mesh.FACEMESH_CONTOURS,
    landmark_drawing_spec=None,
    connection_drawing_spec=mp_drawing_styles
    .get_default_face_mesh_contours_style())
  mp_drawing.draw_landmarks(
      image=image,
      landmark_list=face,
      connections=mp_face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_iris_connections_style())
  if show:
    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    cv2.waitKey(500)

def run():
  # For webcam input:
  cap = cv2.VideoCapture(0)

  tree, labels, known_faces = load_kdtree()
  for known_face in known_faces:
    print("Restoring:", known_face.name, known_face.face_structures.shape[0])
  tracking_faces: list[TrackedFace] = []

  with mp_face_detection.FaceDetection(
      model_selection=0, min_detection_confidence=0.5) as face_detection:
    with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
      with mp_face_mesh.FaceMesh(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
          success, image = cap.read()
          if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

          # To improve performance, optionally mark the image as not writeable to
          # pass by reference.
          image.flags.writeable = False
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = face_detection.process(image)
          hand_results = hands.process(image)
          mesh_results = face_mesh.process(image)


          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          # Draw the face detection annotations on the image.
          if results.detections:
            for detection in results.detections:
              mp_drawing.draw_detection(image, detection)

          # Draw the hand detection annotations on the image
          if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
              mp_drawing.draw_landmarks(
                  image,
                  hand_landmarks,
                  mp_hands.HAND_CONNECTIONS,
                  mp_drawing_styles.get_default_hand_landmarks_style(),
                  mp_drawing_styles.get_default_hand_connections_style())

          # Draw the face mesh annotations on the image
          if mesh_results.multi_face_landmarks:
            def get_face_structure(face):
              lms = np.array([[l.x, l.y, l.z] for l in face.landmark])
              center = np.mean(lms, axis=0)
              size = np.max(lms, axis=0) - np.min(lms, axis=0)
              face_structure = (lms - center) / size
              # pos = (int(lms[0,0] * image.shape[1]), int(lms[0,1] * image.shape[0]))
              # cv2.circle(image, pos, radius=0, color=(0, 0, 255), thickness=1)
              return face_structure

            faces = mesh_results.multi_face_landmarks
            if len(faces) < len(tracking_faces):
              # Lost a face, figure out which one we lost by looking at t-1 since faces don't move much from t-1 to t
              # tracking_faces[face_id].save()
              # del tracking_faces[face_id]
              pass
            elif len(faces) > len(tracking_faces):
              for face_id in range(len(tracking_faces), len(faces), 1):
                face = faces[face_id]
                # Check KDTree to see if face_structure matches close enough
                # Otherwise prompt user to enter the name of the person and start collecting data, save data once we get enough datapoints
                # Show countdown for person to rotate their face
                face_structure = get_face_structure(face=faces[face_id])

                draw_face(image=image, face=face, show=True)
                query_structure = face_structure.reshape((face_structure.shape[0], -1))


                name = input("Enter name:")
                def handle_new_name(name):
                  for tracked_face in tracking_faces:
                    if tracked_face.name == name:
                      # Already tracking someone with this name
                      tracked_face.add_face_structure(face_structure=face_structure, save=False)
                      return
                  for known_face in known_faces:
                    if known_face.name == name:
                      # Restoring someone from the database
                      tracking_faces.append(known_face)
                      known_face.add_face_structure(face_structure=face_structure, save=False)
                      return
                handle_new_name(name)

            for face_id in range(len(faces)):
              face = faces[face_id]
              face_structure = get_face_structure(face=face)
              # Guess who it is, if it's wrong (face_id != pred) then add it to the database
              query_structure = face_structure.reshape((-1,))
              distances, indices = tree.query(query_structure, k=1)
              print(distances, labels[indices], "actual:", labels[face_id])

              tracking_faces[face_id].add_face_structure(face_structure=face_structure, save=False)

              draw_face(image=image, face=face, show=False)

          # Flip the image horizontally for a selfie-view display.
          cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
          if cv2.waitKey(5) & 0xFF == 27:
            break
  for tracked_face in tracking_faces:
    tracked_face.save()

  cap.release()

def main():
  run()

if __name__ == "__main__":
  main()