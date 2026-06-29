# 실시간 지문자 수어 번역기

> Jetson Nano 환경에서 동작하는 실시간 한글 지문자 인식 및 번역 시스템

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![MobileNetV2](https://img.shields.io/badge/MobileNetV2-4285F4?style=flat-square&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=flat-square&logo=google&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat-square&logo=onnx&logoColor=white)
![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Jetson Nano](https://img.shields.io/badge/Jetson%20Nano-76B900?style=flat-square&logo=nvidia&logoColor=white)


## YouTube Demo

아래 이미지를 클릭하면 YouTube 데모 영상으로 이동합니다.
[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo%20Video-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://youtu.be/wPea2OLFxoQ)

<a href="https://youtu.be/wPea2OLFxoQ">
  <img width="720" alt="YouTube Demo" src="https://github.com/user-attachments/assets/6b36b529-acfd-4bc6-9e0f-ef6999ec0719" />
</a>

## 프로젝트 개요

수어가 익숙하지 않은 청각장애인, 중도 실청자, 수어 학습자 등이 지문자를 더 쉽게 이해할 수 있도록 실시간 지문자 수어 번역기를 구현했습니다.

카메라로 입력된 손 이미지를 MediaPipe로 검출하고, MobileNetV2 기반 분류 모델을 통해 한글 지문자 31개 클래스를 예측합니다. 학습된 Keras 모델은 ONNX로 변환한 뒤 Jetson Nano에서 TensorRT 엔진으로 최적화하여 실시간 추론에 사용했습니다.

## 주요 기능

- 실시간 카메라 입력 기반 손 영역 검출
- MediaPipe 기반 손 영역 crop 및 전처리
- MobileNetV2 기반 한글 지문자 31개 클래스 분류
- Top-3 예측 후보 출력
- 예측된 자모를 조합하여 한글 문자열 생성
- 잘못 인식된 음소 수정 UI 제공

- Jetson Nano + TensorRT 기반 실시간 엣지 추론

## Preview

### 학습 및 추론과정

<img width="997" height="560" alt="image" src="https://github.com/user-attachments/assets/fe9d4711-3655-4dd1-90bf-a07bafa2c92e" />

카메라로 입력된 손 이미지를 MediaPipe로 검출하고 MobileNetV2 모델로 지문자를 분류하는 전체 학습 및 추론 흐름입니다.

---

### 모델 구조

<img width="2000" height="1125" alt="model_architecture" src="https://github.com/user-attachments/assets/c83ad26b-3860-4608-a411-aaa901a0b9e4" />

MobileNetV2 위에 Dense Layer를 추가하여 한글 지문자 31개 클래스를 분류하도록 구성한 모델입니다.

---

### 데이터 크롭, 미디어파이프 진행

<img width="333" height="301" alt="image" src="https://github.com/user-attachments/assets/0b9bd354-332c-438c-a387-4ef163579b4a" />

다양한 구도로 촬영한 손 이미지를 MediaPipe로 손 영역을 검출하고 crop하여 학습 데이터로 구축했습니다.

---



## 팀원 역할

| 이름 | 역할 |
|---|---|
| 신성민 | 추론 및 후처리 |
| 정민수 | 학습 및 데이터셋 구축 |
| 조승아 | 학습 및 모델 설계 |
| 여인석 | 추론 및 데이터 수집 |

## 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| Model | MobileNetV2, Transfer Learning |
| Training | TensorFlow, Keras, Google Colab |
| Preprocessing | OpenCV, MediaPipe |
| Optimization | ONNX, TensorRT |
| Edge Device | NVIDIA Jetson Nano |
| UI | Web UI, Camera Streaming |

## 시스템 흐름

```text
Camera Input
-> MediaPipe Hand Detection
-> Hand Crop & Resize
-> MobileNetV2 Inference
-> Top-3 Prediction
-> Post-processing
-> Hangul Text Output
```

## 모델 구조

<img width="941" height="791" alt="image" src="https://github.com/user-attachments/assets/c7f77f59-9a8d-4200-ac77-557433c5dad4" />

MobileNetV2를 기반으로 한글 지문자 31개 클래스를 분류하는 모델을 구성했습니다.

```text
Input Image (224 x 224 x 3)
-> MobileNetV2
-> Global Average Pooling
-> Dense(128)
-> Dropout(0.3)
-> Dense(31)
-> Softmax
```

MobileNetV2는 적은 연산량과 작은 모델 크기를 가지는 경량 CNN 모델입니다. Jetson Nano와 같은 임베디드 환경에서 빠른 추론이 가능하기 때문에 실시간 지문자 예측에 적합하다고 판단했습니다.

## MobileNetV2 선정 이유

YOLO는 이미지 안에서 객체의 위치를 찾는 객체 탐지 모델에 강점이 있습니다. 하지만 본 프로젝트의 핵심은 손의 위치를 찾는 것보다, 추출된 손 이미지가 어떤 지문자인지 분류하는 것입니다.

따라서 손 영역 검출은 MediaPipe로 처리하고, 지문자 분류는 MobileNetV2가 담당하도록 설계했습니다.

```text
YOLO: 객체 위치 탐지에 적합
MobileNetV2: 잘라낸 손 이미지 분류에 적합
```

MobileNetV2는 경량 모델이기 때문에 Jetson Nano에서 실시간으로 동작하기 유리하며, Transfer Learning을 통해 제한된 데이터셋에서도 안정적인 성능을 낼 수 있습니다.

## 데이터셋 구축

<img width="759" height="895" alt="image" src="https://github.com/user-attachments/assets/1ee2dede-49d4-4e65-b30b-14fbabf78f63" />

자음 14개와 모음 17개, 총 31개의 한글 지문자 클래스를 대상으로 데이터셋을 구축했습니다.

초기에는 클래스별 50장에서 100장 사이의 데이터를 수집했으나, 추론 과정에서 일부 클래스의 인식률이 낮아 클래스별 데이터를 150장에서 200장 이상으로 확장했습니다. 이후 일부 데이터는 RGB 변환과 위치 변화 등을 적용하여 약 400장 수준으로 증강했습니다.

데이터 수집 시 손만 타이트하게 크롭한 이미지, 상반신이 배경으로 포함된 이미지, 카메라 전체 프레임 이미지 등 다양한 구도를 의도적으로 포함했습니다.

## 학습 전략

### 데이터 분할

전체 데이터를 7:2:1 비율로 분할했습니다.

```text
학습 데이터 70%: 모델이 지문자 특징 학습
검증 데이터 20%: 학습 중 성능 확인 및 과적합 판단
테스트 데이터 10%: 최종 모델 성능 평가
```

### 데이터 증강

실제 카메라 환경에서는 손의 위치, 각도, 거리, 밝기가 일정하지 않습니다. 따라서 학습 이미지에 회전, 이동, 밝기 변화, 확대/축소를 적용하여 다양한 상황에 대응할 수 있도록 했습니다.

```text
Rotation: 손이 기울어진 경우 대응
Shift: 손 위치가 중앙에서 벗어난 경우 대응
Brightness: 조명 환경 차이 대응
Zoom: 카메라와 손의 거리 차이 대응
```

단, 지문자는 손 방향이 의미를 가지기 때문에 좌우 반전은 적용하지 않았습니다.

## Confusion Matrix

<img width="1975" height="943" alt="image" src="https://github.com/user-attachments/assets/a4b554b3-a6e9-40bf-8bca-e17e1592d5eb" />

모델의 분류 성능을 확인하기 위해 자음과 모음 각각의 혼동행렬을 분석했습니다.

#### 자음 14개 클래스

자음 클래스는 대부분의 예측값이 대각선에 집중되어 있어, 실제 지문자와 예측 지문자가 대부분 일치하는 것을 확인했습니다.

- 대부분의 자음 클래스에서 높은 정확도 확인
- 20회 테스트 기준 대부분 20회 정확히 인식
- 일부 유사한 손 모양에서 소량의 오인식 발생


#### 모음 17개 클래스

모음 클래스 역시 전반적으로 높은 분류 성능을 보였지만, 일부 모음은 손 모양이 비슷해 자음보다 약간의 혼동이 발생했습니다.

- 대부분의 모음 클래스에서 안정적인 인식 결과 확인
- 일부 클래스에서 18~20회 수준의 인식 결과 확인
- 특히 형태가 유사한 모음에서 오인식 가능성 확인



### 결과 요약

혼동행렬 분석 결과, 대부분의 클래스가 실제값과 예측값이 일치하는 대각선에 집중되어 있어 모델이 지문자를 안정적으로 분류하는 것을 확인했습니다.  
다만 손 모양이 유사한 일부 지문자에서는 오인식이 발생하여, 해당 클래스는 추가 데이터 수집과 보강이 필요하다고 판단했습니다.



## ONNX 변환

<img width="2000" height="1125" alt="onnx_flow" src="https://github.com/user-attachments/assets/e368c0a4-3e3e-40c9-a2e7-83d17b1b93e2" />

Keras 모델을 Jetson Nano에서 실행 가능한 형식으로 변환하기 위해 ONNX를 사용했습니다.

```text
Keras 모델
-> ONNX 모델
-> TensorRT 엔진
-> Jetson Nano 실시간 지문자 예측
```

ONNX 변환을 통해 Jetson Nano 실행 환경과의 호환성을 확보하고, TensorRT 최적화를 적용하여 실시간 추론 속도를 향상시켰습니다.

## 추론 파이프라인

<img width="1007" height="571" alt="image" src="https://github.com/user-attachments/assets/244b53a4-cc12-4158-bc84-e1144fd477b6" />

카메라에서 입력된 320 x 240 RGB 이미지를 초당 30프레임으로 받아 처리합니다.

전처리는 두 단계로 구성했습니다.

```text
1. MediaPipe로 손 위치와 크기 검출
2. 검출된 손 영역 crop 후 224 x 224 RGB 이미지로 변환
```

모델은 224 x 224 x 3 형태의 이미지를 입력으로 받고, 31개 지문자 클래스에 대한 확률값을 출력합니다.

## 후처리 및 UI
<img width="2000" height="1125" alt="ui_design" src="https://github.com/user-attachments/assets/9394bf5b-963e-4844-8083-7a4d2965fc37" />
<img width="2000" height="1125" alt="preprocessing" src="https://github.com/user-attachments/assets/eecb5165-5ca7-4e64-bf02-b73bc3cb3dea" />


모델 출력 결과 중 인식률이 높은 상위 3개의 음소를 UI에 표시합니다. 가장 높은 확률의 음소는 자모 큐에 저장되며, 저장된 자모를 조합하여 한글 문자열로 출력합니다.
UI는 웹 브라우저에서 접속할 수 있도록 구성했으며, 다음 기능을 제공합니다.

- 카메라 이미지 출력
- 손 인식 영역 표시
- 인식된 자모 및 문자열 출력
- Top-3 후보 선택
- 잘못 인식된 음소 수정
- 입력 문장 초기화

## Trouble Shooting

### 1. 학습 데이터 수 부족 문제

초기 학습에서는 일부 클래스에서 인식률이 낮게 나타났습니다. 데이터 수가 부족하다고 판단하여 클래스별 데이터를 150장에서 200장 이상으로 확장하고, RGB 변환과 위치 변화 등을 적용해 약 400장 수준으로 증강했습니다.

하지만 데이터 수를 늘린 뒤에도 동일한 추론 오류가 발생했습니다. 이를 통해 단순히 데이터 수만 늘리는 것보다 학습 데이터와 실제 추론 입력의 형태를 일치시키는 것이 더 중요하다는 점을 확인했습니다.

### 2. 학습/추론 입력 형태 불일치

학습 단계에서는 원본 이미지 기반으로 학습했지만, 추론 단계에서는 MediaPipe로 손 영역을 crop한 이미지가 모델 입력으로 사용되었습니다.

```text
학습 입력: 원본 이미지 중심
추론 입력: MediaPipe crop 이미지
```

두 입력 형태가 달라 모델이 실제 카메라 입력에서 안정적으로 예측하지 못하는 문제가 발생했습니다. 이후 학습 데이터도 MediaPipe 기반으로 손 영역을 crop하여 추론 입력과 동일한 방식으로 맞춘 뒤 인식률이 개선되었습니다.

### 3. 유사 지문자 오인식 문제
<img width="1001" height="555" alt="image" src="https://github.com/user-attachments/assets/25042bdd-950d-4119-99ff-ef4b0d1ef3ec" />

일부 지문자는 손 모양이 유사하여 Top-1 결과의 신뢰도가 낮거나 Top-2에 정답이 포함되는 경우가 발생했습니다. 이를 해결하기 위해 Top-3 후보를 UI에 표시하고, 사용자가 잘못 인식된 음소를 수정할 수 있도록 구성했습니다.

### 4. 자모 조합 후처리 문제
<img width="984" height="551" alt="image" src="https://github.com/user-attachments/assets/3d14d92f-c33c-444d-8472-77bdd694d5e7" />

지문자는 자음과 모음이 순서대로 입력되기 때문에 음절 조합 규칙이 필요했습니다. 자음 이후 모음이 입력되면 하나의 음절로 조합하고, 종성으로 처리된 자음 뒤에 모음이 입력되면 음절을 분리하는 방식으로 후처리를 설계했습니다.

## 결과
<img width="964" height="554" alt="image" src="https://github.com/user-attachments/assets/54708492-144f-4827-9887-bf7abe345d23" />
<img width="745" height="420" alt="image" src="https://github.com/user-attachments/assets/8420c15f-af18-4410-9548-a9759e060f23" />

학습 데이터와 추론 입력 방식을 동일하게 맞춘 뒤, 실제 카메라 입력에서도 의도한 지문자를 예측할 수 있음을 확인했습니다.

자음과 모음 클래스 대부분에서 안정적인 인식 결과를 보였으며, 일부 혼동이 발생하는 클래스는 데이터 보강과 UI 후보 선택 기능을 통해 보완했습니다.

## 후속 연구

- 의미 기반 수어 인식으로 확장
- 인식률이 낮은 클래스의 데이터 추가 수집
- 사용자별 손 크기와 촬영 환경 차이에 대응하는 일반화 성능 개선
- 음절 조합 알고리즘 고도화
- Jetson Nano 환경에서 추론 속도 및 UI 반응성 최적화

## 느낀 점

이번 프로젝트를 통해 모델 구조만큼이나 데이터 품질과 전처리 방식이 성능에 큰 영향을 준다는 점을 체감했습니다. 같은 지문자라도 사람마다 손 크기, 손 모양, 각도, 조명, 카메라 위치가 달라지기 때문에 안정적인 모델을 만들기 위해서는 다양한 조건의 데이터가 필요했습니다.

또한 수어는 단순한 손동작이 아니라 청각장애인에게 중요한 의사소통 수단이라는 점을 다시 생각하게 되었습니다. 기술적으로 정확도를 높이는 것도 중요하지만, 실제 사용자가 더 편하게 소통할 수 있도록 돕는 방향으로 개발하는 것이 중요하다고 느꼈습니다.

## Thanks


<img width="544" height="306" alt="thanks" src="https://github.com/user-attachments/assets/c3bd8318-04ba-4921-8fb0-f823b7276eae" />
