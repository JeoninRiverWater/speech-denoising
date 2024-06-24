import torch
from torch.utils.data import DataLoader, Dataset

# 예시 데이터셋 정의
class MyDataset(Dataset):
    def __init__(self):
        self.data = [i for i in range(100)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main() : 
    # 데이터셋 인스턴스 생성
    dataset = MyDataset()

    # DataLoader 인스턴스 생성
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # DataLoader 사용 예시
    for batch in dataloader:
        print(batch)

if __name__ == '__main__'  : 
    main()