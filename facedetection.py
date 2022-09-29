import cv2
import mediapipe as mp
import time

class face_detection:
    def __init__(self,min_detection_confidence=0.75, model_selection=0):
        self.min_detection_confidence=min_detection_confidence
        self.model_selection=model_selection
        
        self.mpface =  mp.solutions.mediapipe.python.solutions.face_detection
        self.face = self.mpface.FaceDetection(self.min_detection_confidence,self.model_selection)
        self.mpdraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        
    def findface(self,img,draw=True):
        self.result = self.face.process(img)
        bbox=[]
        if self.result.detections:
            for id , detection in enumerate(self.result.detections):
                # mpdraw.draw_detection(img,detection)
                score=round(detection.score[0]*100,2)
                box = detection.location_data.relative_bounding_box
                h,w,c=img.shape
                x,y,w,h=int(box.xmin*w),int(box.ymin*h),int(box.width*w),int(box.height*h)
                bbox.append([id,score,box])
                if draw:
                    img=self.fancydraw(img,box)
                    cv2.putText(img,f'{score}%',(x,y-10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),1)
        return img,bbox
    def fancydraw(self,img,box,l=30,t=5,rt=1):
        h,w,c=img.shape
        x,y,w,h=int(box.xmin*w),int(box.ymin*h),int(box.width*w),int(box.height*h)
        x1,y1=x+w,y+h
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),rt)
        #Top Left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)
        #Top Right x,y
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)
        #Bottom Left x,y
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)
        #Bottom Right x,y
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)
        return img

def main():
    cap = cv2.VideoCapture(0)
    ptime=0
    ctime=0
    detector = face_detection()
    while True:
        status,img = cap.read()
        img,box=detector.findface(img)
        if len(box)!=0:
            print(box[0])
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        
        cv2.putText(img,f'FPS {str(int(fps))}',(10,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__=="__main__":
    main()