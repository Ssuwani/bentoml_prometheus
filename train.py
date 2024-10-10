import bentoml
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(
                28 * 28, 512
            ),  # In: 28*28, Out: 512 (28*28의 이미지가 Flatten 되어 512개의 텐서가 입력)
            nn.ReLU(),
            nn.Linear(512, 512),  # In: 512, Out: 512
            nn.ReLU(),
            nn.Linear(512, 10),  # In: 512, Out: 10 -> 10개의 카테고리로 분류
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)  # 10개의 텐서 = 각 클래스에 대한 확률
        return logits


model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    for batch, (x, y) in enumerate(train_dataloader):
        pred = model(x)  # 순전파
        loss = loss_fn(pred, y)  # Loss 계산
        loss.backward()  # 역전파
        optimizer.step()  # 매개변수 업데이트
        optimizer.zero_grad()  # 그레이디언트 초기화
        if batch % 100 == 0:  # 100번째 배치마다 Loss 출력
            print(f"loss: {loss.item():>7f}")

model.eval()  # 모델을 평가 모드로 설정
correct = 0
with torch.no_grad():
    for x, y in test_dataloader:
        pred = model(x)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

correct /= len(test_dataloader.dataset)
print(f"Accuracy: {(100*correct):>0.1f}%")

# BentoML로 모델 저장
bentoml.pytorch.save_model(
    "fashion_mnist_model",
    model,
    labels={"framework": "pytorch"},
    metadata={
        "accuracy": correct,
        "author": "jangsuwan",
    },
    custom_objects={
        "labels": [
            "T-shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    },
)
