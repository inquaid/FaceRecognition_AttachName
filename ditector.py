from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import argparse
import cv2
import numpy as np

# from PIL.ImageFont import ImageFont
from PIL import ImageFont

DEFAULT_ENCODING_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "--camera", action="store_true", help="Use camera to detect faces"
)
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog(CPU), cnn(GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def encode_known_faces(
        model: str = "hog", encodings_location: Path = DEFAULT_ENCODING_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


# encode_known_faces()


def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODING_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )

    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
            input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)
        # print(name, bounding_box)
    del draw
    pillow_image.show()


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


# recognize_faces("unknown.jpg")

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", 40)
    except:
        font = ImageFont.load_default()
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name, font=font
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white", font=font,
    )


def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )


# validate()


def recognize_faces_in_camera(
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODING_PATH
) -> None:
    try:
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)
        print(f"Loaded encodings for {len(set(loaded_encodings['names']))} people")
    except FileNotFoundError:
        print(f"Error: Train first! No encodings at {encodings_location}")
        return
    except Exception as e:
        print(f"Error loading encodings: {str(e)}")
        return

    video_capture = cv2.VideoCapture(0)
    for _ in range(3):
        if video_capture.isOpened():
            break
        video_capture.open(0)

    if not video_capture.isOpened():
        print("Error: Camera not accessible. Check permissions or try different camera index.")
        return

    print("Camera active. Press Q to quit...")
    process_frame = True
    detection_debug = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Frame capture error")
            break

        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            number_of_times_to_upsample=1,
            model=model
        )

        if detection_debug:
            print(f"Detected {len(face_locations)} faces in frame")

        if len(face_locations) == 0 and detection_debug:
            cv2.imwrite("debug_no_face.jpg", frame)
            print("Saved debug_no_face.jpg for analysis")

        recognized_names = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    loaded_encodings["encodings"],
                    face_encoding,
                    tolerance=0.55
                )

                votes = Counter(
                    name for match, name in zip(matches, loaded_encodings["names"])
                    if match
                )

                if votes:
                    name, count = votes.most_common(1)[0]
                    recognized_names.append(name if count >= 2 else "Unknown")
                else:
                    recognized_names.append("Unknown")

        for (top, right, bottom, left), name in zip(face_locations, recognized_names):
            scale = int(1 / scale_factor)
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)

            cv2.putText(frame, name,
                        (left + 10, bottom - 10),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.0, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.camera:
        recognize_faces_in_camera(model=args.m)
