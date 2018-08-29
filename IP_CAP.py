import cv2
import sys

def camera():
    """input: None
       output: the videocapture class"""
    gst = "rtspsrc location=rtsp://admin:12345@192.168.1.64:554/h264/ch1/main/av_stream latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink"
    cap = cv2.VideoCapture(gst)
    return cap


if __name__ == "__main__":
    cap = camera()
    if not cap.isOpened():
        sys.exit("Failed to open camera!")
    succes, frame = cap.read()
    while succes:
        cv2.imshow('video', frame)
        succes, frame = cap.read()
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            break
    
    cap.release()
    cv2.destroyAllWindows()
