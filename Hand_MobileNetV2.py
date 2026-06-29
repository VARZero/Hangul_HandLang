#######################################
# 코드는 같은데 모델만 다르다.

# main.py
import numpy as np
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import pycuda.autoinit  # PyCUDA 컨텍스트 자동 관리용 수입

# import TRTInferenceEngine
from trt_module import TRTInferenceEngine

# 초기 상숫값 정의
# ENGINE_PATH = 'RPS_MobileNetV2_Augmentation.sim.engine'
ENGINE_PATH = 'Korean_Fingerspelling_MobileNetV2.sim.engine'
IMG_SIZE = 224
OFFSET = 15
# CAMW = 640
# CAMH = 480
CAMW = 320
CAMH = 240

# 손 검출 디텍터 및 TensorRT 가속 클래스 초기화
hd = HandDetector(maxHands=1)
trt_engine = TRTInferenceEngine(ENGINE_PATH)

# ansToText = {0: 'scissors', 1: 'rock', 2: 'paper'}
LABELS = [
    "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ",
    "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    "ㅏ", "ㅐ", "ㅑ", "ㅒ",
    "ㅓ", "ㅔ", "ㅕ", "ㅖ",
    "ㅗ", "ㅚ", "ㅛ",
    "ㅜ", "ㅟ", "ㅠ",
    "ㅡ", "ㅢ", "ㅣ"
]
ansToText = {i: label for i, label in enumerate(LABELS)}

colorList = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def make_square_img(img):
    ho, wo = img.shape[0], img.shape[1]
    aspectRatio = ho / wo
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    
    if aspectRatio > 1:  # Portrait 구조
        k = IMG_SIZE / ho
        wk = int(wo * k)
        img = cv2.resize(img, (wk, IMG_SIZE))
        img_h, img_w = img.shape[0], img.shape[1]
        d = (IMG_SIZE - img_w) // 2
        wbg[:img_h, d:img_w + d] = img
    else:  # Landscape 구조
        k = IMG_SIZE / wo
        hk = int(ho * k)
        img = cv2.resize(img, (IMG_SIZE, hk))
        img_h, img_w = img.shape[0], img.shape[1]
        d = (IMG_SIZE - img_h) // 2
        wbg[d:img_h + d, :img_w] = img
    return wbg

def processImage(frame):
    # 1. 손 검출 시도
    hands, _ = hd.findHands(frame, draw=False)
    if not hands:
        return

    # Bounding Box 정보 획득
    x, y, w, h = hands[0]['bbox']

    # 프레임 경계 범위 초과 방지 예외 처리
    # if x < OFFSET or y < OFFSET or x + w + OFFSET > 320 or y + h > 240:
    #     return

    # ROI 영역 추출을 위한 좌표 보정
    # x1, y1 = x - OFFSET, y - OFFSET
    # x2, y2 = x + w + OFFSET, y + h
    x1, y1 = max(x - OFFSET, 0), max(y - OFFSET, 0)
    x2, y2 = min(x + w + OFFSET, CAMW), min(y + h, CAMH)

    # 2. 전처리 (손 영역 Crop -> 정형화 -> 타입 캐스팅)
    img = frame[y1:y2, x1:x2]
    img = make_square_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    inp = np.expand_dims(img, 0)  # (1, 224, 224, 3)

    # # 3. 추론 실행
    # output_host = trt_engine.infer(inp)

    # # 4. 결과 해석 및 바인딩 박스 연출
    # output_data = output_host.ravel()
    # ans = int(np.argmax(output_data))
    # text = ansToText.get(ans, str(ans))

    # cv2.rectangle(frame, (x1, y1), (x2, y2), colorList[ans], 2)
    # cv2.putText(frame, text, (x1, y1 - 7), cv2.FONT_HERSHEY_PLAIN, 2, colorList[ans], 2)

    # 3. 추론 실행
    output_host = trt_engine.infer(inp)

    # 4. 결과 해석
    output_data = output_host.ravel()

    pred_idx = int(np.argmax(output_data))
    pred_prob = float(output_data[pred_idx])

    print(pred_idx, ansToText[pred_idx], pred_prob)

    text = f"{ansToText[pred_idx]} ({pred_prob:.2f})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        text,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


# V4L2 카메라 인터페이스 초기화 및 환경 세팅
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('cam', 320 + 40, 240 + 60)
cv2.resizeWindow('cam', CAMW + 40, CAMH + 60)

startTime = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: 
        break

    # 프레임 단위 처리 실행
    processImage(frame)

    # FPS 실시간 계산 연출
    curTime = time.time()
    fps = 1 / (curTime - startTime)
    startTime = curTime
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    cv2.imshow('cam', frame)
    if cv2.waitKey(10) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()