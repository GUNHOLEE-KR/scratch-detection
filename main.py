import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def main():
    print("=== [비전 검사] 원통 표면 스크래치 탐지 AI 초정밀 학습 시작 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 현재 사용 중인 하드웨어 장치: {device}")

    # 흑백 이미지 특화 전처리 (선명도 및 명암비 고정 버프)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),          
            transforms.ColorJitter(brightness=0.1, contrast=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    try:
        image_datasets = {x: datasets.ImageFolder(os.path.join(BASE_DIR, x), data_transforms[x]) for x in ['train', 'test']}
    except FileNotFoundError:
        print("\n[오류] 데이터 폴더를 찾을 수 없습니다.")
        return

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=8, shuffle=(x == 'train'), num_workers=0)
        for x in ['train', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names   = image_datasets['train'].classes

    train_targets = image_datasets['train'].targets
    class_counts  = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ResNet50 체급 유지
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 2)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # [변경] CPU 환경에서 빠르고 정밀하게 수렴하는 AdamW 옵티마이저 채택
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.00005,                 # 안정적인 미세 조정을 위해 학습률을 최적화
        weight_decay=1e-3
    )

    # 60에폭에 맞춰 부드럽게 감소하는 코사인 스케줄러
    epochs = 60                      
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n Epoch {epoch+1:02d} / {epochs}  |  LR: {current_lr:.6f}")
        print(f"{'─'*50}")

        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            tag = "★ BEST" if (phase == 'test' and epoch_acc > best_acc) else ""
            print(f" [{phase.upper():5s}]  Loss: {epoch_loss:.4f}  |  Acc: {epoch_acc:.4f}  {tag}")

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

    SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch_resnet18_best.pth')
    torch.save(best_weights, SAVE_PATH)
    print(f"\n 최고 검증 정확도: {best_acc:.4f} ➔ 모델 저장 완료")

    # 최종 리포트 출력
    model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print("\n 혼동행렬 (Confusion Matrix):")
    print(f"              예측: good  예측: scratch")
    print(f"  실제: good     {cm[0][0]:5d}       {cm[0][1]:5d}")
    print(f"  실제: scratch  {cm[1][0]:5d}       {cm[1][1]:5d}")
    print("\n 상세 성능 리포트:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

if __name__ == '__main__':
    main()
