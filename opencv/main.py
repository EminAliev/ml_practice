import cv2

if __name__ == "__main__":
    model = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(model)
    input_image = cv2.imread('input.jpg')

    scale_coefficient = 0.25
    width = int(input_image.shape[1] * scale_coefficient)
    height = int(input_image.shape[0] * scale_coefficient)
    dimensions = (width, height)

    resized = cv2.resize(input_image, dimensions, interpolation=cv2.INTER_AREA)
    gray_scaled_color = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    for (x, y, width, height) in face_cascade.detectMultiScale(gray_scaled_color, 1.1, 12):
        red_color = (0, 0, 255)
        cv2.rectangle(resized, (x, y), (x + width, y + height), red_color, 2)

    cv2.imshow('result', resized)
    cv2.waitKey()

