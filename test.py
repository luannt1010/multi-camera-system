import threading
import cv2


def PlayCamera(id):
    video_capture = cv2.VideoCapture(id)
    while True:

        ret, frame = video_capture.read()
        cv2.imshow('{}'.format(id), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()


url1 = "rtsp://localhost:8554/live.sdp"
url2 = "rtsp://admin:hd543211@192.168.1.127:554/0"
cameraIDs = [0, url1]

for id in cameraIDs:
    t = threading.Thread(target=PlayCamera, args=(id,))
    t.start()