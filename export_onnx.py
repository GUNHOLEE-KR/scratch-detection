# export_onnx.py (전체 교체용 파이썬 코드)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet50WithCAM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        # 마지막 특징 맵을 추출하기 위해 레이어 분할
        self.features = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
            base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
        )
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

    def forward(self, x):
        # 1. 마지막 특징 맵 추출 (Shape: [1, 2048, 7, 7])
        feat = self.features(x)
        
        # 2. 기존 분류 결과 계산
        pooled = self.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        output = self.fc(flat)
        
        # 3. 불량(scratch) 클래스에 해당하는 채널 가중치 추출 (fc 레이어의 가중치 활용)
        # scratch_weights shape: [2048]
        scratch_weights = self.fc[1].weight[1]
        
        # 4. 특징 맵에 불량 가중치를 곱해 불량 근거 맵(Heatmap) 생성
        # feat를 [2048, 49]로 펼치고 가중치와 행렬 연산 후 다시 [7, 7]로 복원
        b, c, h, w = feat.shape
        feat_flat = feat.view(c, h * w)
        cam_map = torch.matmul(scratch_weights, feat_flat)
        cam_map = cam_map.view(1, 1, h, w)
        
        # 5. 입력 이미지 크기(224x224)로 해상도 업샘플링
        cam_map = F.interpolate(cam_map, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 분류 결과(output)와 불량 영역 지도(cam_map)를 동시에 반환
        return output, cam_map

def load_model():
    base_model = models.resnet50(weights=None)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 2)
    )
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PTH_PATH = os.path.join(BASE_DIR, 'scratch_resnet18_best.pth')
    
    print(f" .pth 로드 중: {PTH_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.load_state_dict(torch.load(PTH_PATH, map_location=device))
    
    # CAM 출력이 결합된 커스텀 모델로 래핑
    model = ResNet50WithCAM(base_model)
    model.to(device)
    model.eval()
    return model, device

def main():
    print("=== CAM 시각화 지원 ONNX 모델 변환 시작 ===")
    model, device = load_model()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ONNX_PATH = os.path.join(BASE_DIR, 'scratch_resnet18_best.onnx')
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    print(" ONNX 변환 중...")
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,        
        opset_version=18,          
        do_constant_folding=True,  
        input_names=['input'],     
        output_names=['output', 'cam'], # C#에서 읽을 두 개의 출력 지정
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}, 'cam': {0: 'batch_size'}} 
    )
    print(f" 변환 완료! ONNX 파일 생성됨 ➔ {ONNX_PATH}")

if __name__ == '__main__':
    main()
