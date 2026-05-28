import os

folders = [
    'data/train/good',
    'data/train/scratch',
    'data/test/good',
    'data/test/scratch',
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f'생성 완료: {folder}')

print('\n폴더 생성 완료!')
print('각 폴더에 이미지를 넣고 main.py를 실행하세요.')
