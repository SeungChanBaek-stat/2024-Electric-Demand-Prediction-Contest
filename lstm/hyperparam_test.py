import time
import torch
from torch.utils.data import DataLoader

def measure_loading_time(dataset, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    start_time = time.time()
    for i, (inputs, targets) in enumerate(loader):
        if i >= 10:  # 10번의 배치 로딩 시간 측정
            break
    end_time = time.time()
    return end_time - start_time

# 가상의 데이터셋
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length):
        self.length = length
        self.data = torch.randn(length, 3, 128, 100)
        self.targets = torch.randint(0, 10, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.freeze_support()

    dataset = DummyDataset(40000)

    batch_size = 256
    for num_workers in [1, 2, 3, 4, 5, 6, 7, 8]:
        loading_time = measure_loading_time(dataset, batch_size, num_workers)
        print(f"Num workers: {num_workers}, Loading time: {loading_time:.4f} seconds")