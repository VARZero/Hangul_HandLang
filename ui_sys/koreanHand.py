import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from cvzone.HandTrackingModule import HandDetector
import pycuda.driver as cuda
import pycuda.autoinit
import copy
import threading
import time  # 스레드 스케줄링을 위해 임포트

from trt_module import TRTInferenceEngine
from hangelJAMO import hangelAlphabetQueue

app = Flask(__name__)

# 설정 및 초기화
ENGINE_PATH = './Hand_MobileNetV2_Augmentation.sim.engine'
IMG_SIZE = 224
OFFSET = 32
CAMW = 320
CAMH = 240

hd = HandDetector(maxHands=1)
trt_engine = TRTInferenceEngine(ENGINE_PATH)

ansToText = { 
    0:"ㄱ", 1:"ㄴ", 2:"ㄷ", 3:"ㄹ", 4:"ㅁ", 5:"ㅂ", 6:"ㅅ", 7:"ㅇ", 8:"ㅈ", 9:"ㅊ", 
    10:"ㅋ", 11:"ㅌ", 12:"ㅍ", 13:"ㅎ", 14:"ㅏ", 15:"ㅑ", 16:"ㅓ", 17:"ㅕ", 18:"ㅗ", 
    19:"ㅛ", 20:"ㅜ", 21:"ㅠ", 22:"ㅡ", 23:"ㅣ", 24:"ㅐ", 25:"ㅒ", 26:"ㅔ", 27:"ㅖ", 
    28:"ㅚ", 29:"ㅟ", 30:"ㅢ" 
}

# 스레드 안전을 위한 락과 자모 큐
jamo_queue = hangelAlphabetQueue()
data_lock = threading.Lock()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMH)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 초기화 시점의 메인 CUDA 컨텍스트 미리 저장
main_cuda_ctx = cuda.Context.get_current()

def make_square_img(img):
    ho, wo = img.shape[0], img.shape[1]
    aspectRatio = ho / wo
    wbg = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
    if aspectRatio > 1:
        k = IMG_SIZE / ho
        wk = int(wo * k)
        img = cv2.resize(img, (wk, IMG_SIZE))
        wbg[:img.shape[0], ((IMG_SIZE - img.shape[1]) // 2):img.shape[1] + ((IMG_SIZE - img.shape[1]) // 2)] = img
    else:
        k = IMG_SIZE / wo
        hk = int(ho * k)
        img = cv2.resize(img, (IMG_SIZE, hk))
        wbg[((IMG_SIZE - img.shape[0]) // 2):img.shape[0] + ((IMG_SIZE - img.shape[0]) // 2), :img.shape[1]] = img
    return wbg

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def processImage(frame):
    hands, _ = hd.findHands(frame, draw=False)
    if not hands: return None, None
    x, y, w, h = hands[0]['bbox']
    x1, y1 = max(x - OFFSET, 0), max(y - OFFSET, 0)
    x2, y2 = min(x + w + OFFSET, CAMW), min(y + h + OFFSET, CAMH)
    img = frame[y1:y2, x1:x2]
    if img.size == 0: return None, None
    img = make_square_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    inp = np.expand_dims(img, 0)
    
    # 순수 추론 부분만 위아래에서 컨텍스트 제어
    main_cuda_ctx.push()
    try:
        output_host = trt_engine.infer(inp)
        ans = int(np.argmax(output_host.ravel()))
    finally:
        main_cuda_ctx.pop()
        
    return ans, (x, y, w, h)


def generate_frames():
    global jamo_queue
    queue = []
    lastChar = ''
    missed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        ans, bbox = processImage(frame)

        if bbox is not None:
            missed_frames = 0
            x, y, w, h = bbox
            draw_margin = 10
            x1_draw = max(x - draw_margin, 0)
            y1_draw = max(y - draw_margin, 0)
            x2_draw = min(x + w + draw_margin, CAMW)
            y2_draw = min(y + h + draw_margin, CAMH)
            
            # 예측값을 분류 큐에 즉시 추가
            queue.append(ans)
            if len(queue) > 20:
                queue.pop(0)
                
            # 큐 내 자모 일치율을 기준으로 안정성(시각 피드백) 및 입력 처리
            is_stable = False
            if len(queue) >= 20:
                ansDone = max(set(queue), key=queue.count)
                count = queue.count(ansDone)
                MIN_RATIO_COUNT = 10  # 55% 이상 일치 시 안정(Green)으로 판단
                
                if count >= MIN_RATIO_COUNT:
                    is_stable = True
                    if ansDone != lastChar:
                        char = ansToText[ansDone]
                        print(f"새로운 음소 추가됨: {char} (비율 만족: {count}/20)")
                        
                        with data_lock:
                            jamo_queue + char
                        
                        lastChar = ansDone
            
            if is_stable:
                # 안정 상태: 초록색 굵은 상자
                cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 2)
            else:
                # 불안정/과도기 상태: 주황색 얇은 상자
                cv2.rectangle(frame, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 165, 255), 1)
        else:
            missed_frames += 1
            if missed_frames > 15:
                queue.clear()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        

@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>수어 인식 시스템 UI</title>
        <style>
            body { font-family: sans-serif; background-color: #fff; margin: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; }
            .container { text-align: center; width: 100%; max-width: 640px; }
            .video-box img { width: 100%; max-width: 480px; height: auto; border: 1px solid #eaeaea; box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.05); }
            .text-stream-container { width: 85%; margin: 30px auto; border-bottom: 2px solid #222; padding-bottom: 8px; text-align: left; }
            .text-stream { font-size: 46px; font-weight: bold; min-height: 55px; display: flex; align-items: center; }
            .text-stream span.last-letter { background-color: #1a4386; color: #ffffff; padding: 0 6px; margin-left: 2px; }
            .btn-reset { background-color: #fff; color: #000; border: 2px solid #1a4386; font-size: 22px; font-weight: bold; padding: 12px 65px; cursor: pointer; letter-spacing: 6px; }
            .btn-reset:hover { background-color: #1a4386; color: #fff; }
        </style>
    </head>
    <body>
    <div class="container">
        <div class="video-box"><img src="{{ url_for('video_feed') }}"></div>
        <div class="text-stream-container"><div class="text-stream" id="result-text">&nbsp;</div></div>
        <button class="btn-reset" onclick="resetText()">초 기 화</button>
    </div>
    <script>
        function fetchText() {
            fetch('/get_text')
                .then(res => res.json())
                .then(data => {
                    const el = document.getElementById('result-text');
                    el.innerHTML = (data.html_content && data.html_content.trim() !== "") ? data.html_content : "&nbsp;";
                }).catch(err => console.log(err));
        }
        function resetText() {
            fetch('/reset', { method: 'POST' }).then(() => { document.getElementById('result-text').innerHTML = "&nbsp;"; });
        }
        setInterval(fetchText, 200); // 0.2초마다 갱신
    </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global jamo_queue
    
    with data_lock:
        temp_queue = copy.deepcopy(jamo_queue)
        
    assembled_text = temp_queue.gatherUMJEOAL()
    
    if len(assembled_text) > 0:
        base_text = assembled_text[:-1]
        last_char = assembled_text[-1]
        html_content = f"{base_text}<span class='last-letter'>{last_char}</span>"
    else:
        html_content = "&nbsp;"
        
    return jsonify(html_content=html_content)

@app.route('/reset', methods=['POST'])
def reset():
    global jamo_queue
    with data_lock:
        jamo_queue.q.clear()
    return jsonify(status="success")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)