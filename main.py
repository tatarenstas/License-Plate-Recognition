import cv2
from ultralytics import YOLO
from utils import read_license_plate

license_plate_detector = YOLO('license_plate_detector.pt')

def get_license_plate(image):
    license_plates = license_plate_detector(image)[0]
    numbers = []
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
        numbers.append([x1, y1, x2, y2, license_plate_text])
        
    return numbers

def get_license_plate_from_video(path_file):
    cap = cv2.VideoCapture(f"{path_file}")
    while True:
        ret, frame = cap.read()
        numbers = get_license_plate(frame)
        for number in numbers:
            x1, y1, x2, y2, license_plate_text = number
            if license_plate_text != None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv2.putText(img = frame, text = f"{license_plate_text}", org = (x1, y1), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 3.0, color = (0, 255, 0), thickness = 2)
        scale_percent = 30
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) % 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_license_plate_from_image(path_file):
    numbers = []
    image = cv2.imread(f"{path_file}")
    license_plate = get_license_plate(image)
    for plate in license_plate:
        x1, y1, x2, y2, license_plate_text = plate
        numbers.append(license_plate_text)
    print(numbers)

get_license_plate_from_image('test.jpg')