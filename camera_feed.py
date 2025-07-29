import cv2
import logging
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, src=0):
        logger.info(f"Initializing VideoStream with source: {src}")
        self.cap = cv2.VideoCapture(src) #initialize web cam
        logger.info("VideoCapture object created")
        self.ret, self.frame = self.cap.read() #read first frame
        logger.info(f"First frame read result: {self.ret}")
        self.stopped = False #flag to stop thread
        logger.info("Starting background thread for frame updates")
        Thread(target=self.update, args=()).start() #background thread to read frames from camera

    def update(self): #method to read frames from camera
        logger.info("Background update thread started")
        frame_count = 0
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read() #read frame from camera
                frame_count += 1
                # Only log every 30 frames (about once per second at 30fps)
                if frame_count % 30 == 0:
                    logger.info(f"Background thread: Read {frame_count} frames, latest result: {self.ret}")
    
    def read(self):
        logger.info("Returning latest frame")
        return self.ret, self.frame #return the latest frame

    def stop(self):
        logger.info("Stopping video stream")
        #stop the video stream and release camera
        self.stopped = True
        logger.info("Stopped flag set to True")
        self.cap.release()
        logger.info("Camera released")

    def __del__(self):
        logger.info("VideoStream destructor called")
        self.stop()

