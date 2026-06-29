# trt_module.py
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class TRTInferenceEngine:
    def __init__(self, engine_path):
        """
        TensorRT 10.x 최신 표준에 맞춘 고성능 추론 엔진 모듈
        """
        # 1. 내부 로거 및 런타임 초기화
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 2. 정적 엔진 가중치 파일 역직렬화(Deserialize)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        # 3. 동적 연산 상태 공간(Context) 및 비동기 처리용 CUDA 스트림 생성
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 입출력 버퍼 메타데이터 보관소
        self.tensors = {}

        # 4. 하드웨어 주소 자동 매핑 구조 설정
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.context.get_tensor_shape(tensor_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            mode = self.engine.get_tensor_mode(tensor_name)
            
            # CPU-GPU 하드웨어 장치가 물리 메모리를 공유하는 Zero-Copy 버퍼 할당
            host_mem = cuda.pagelocked_empty(size, dtype, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
            device_ptr = host_mem.base.get_device_pointer()
            
            # [공식 API] 하드웨어 버스 주소를 이름 기준으로 텍스트 다이렉트 매핑
            self.context.set_tensor_address(tensor_name, int(device_ptr))
            
            self.tensors[tensor_name] = {
                'host': host_mem,
                'shape': shape,
                'mode': mode
            }
            
        # 단일 입출력 구조 모델을 위한 이름 가독성 처리
        self.input_name = [name for name, info in self.tensors.items() if info['mode'] == trt.TensorIOMode.INPUT][0]
        self.output_name = [name for name, info in self.tensors.items() if info['mode'] == trt.TensorIOMode.OUTPUT][0]

    def infer(self, input_data):
        """
        메모리 호스트 복사 부하가 전혀 없는 순수 비동기 가속 추론 메서드
        """
        # 1. 전처리된 데이터를 CPU 공유 버퍼에 타겟팅 (하드웨어 매핑으로 GPU에 즉시 연동)
        np.copyto(self.tensors[self.input_name]['host'].reshape(self.tensors[self.input_name]['shape']), input_data)
        
        # 2. 공식 API 구동 (명시적인 bindings 배열 포인터를 던지지 않고 실행)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 3. 비동기 GPU 하드웨어 연산 대기 동기화
        self.stream.synchronize()
        
        # 4. 출력용 CPU 매핑 버퍼 핸들 반환
        return self.tensors[self.output_name]['host']