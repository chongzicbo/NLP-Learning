import torch
import copy
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader
import accelerate

# seed
set_seed(0)

# define toy inputs and labels
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
gradient_accumulation_steps = 4
batch_size = len(x) // gradient_accumulation_steps

# define dataset and dataloader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size)

# define model, optimizer and loss function
model = torch.zeros((1, 1), requires_grad=True)

model_clone = copy.deepcopy(model)

criterion = torch.nn.MSELoss()
model_optimizer = torch.optim.SGD([model], lr=0.02)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
device = accelerator.device
model, model_optimizer, dataloader = accelerator.prepare(
    model, model_optimizer, dataloader
)
model = model.to(device)
# model_clone = model_clone.to(device)
model_clone_optimizer = torch.optim.SGD([model_clone], lr=0.02)
print(f"initial model weight is {model.mean().item():.5f}")
print(f"initial model weight is {model_clone.mean().item():.5f}")
for i, (inputs, labels) in enumerate(dataloader):
    with accelerator.accumulate(model):
        inputs = inputs.view(-1, 1)
        print(i, inputs.flatten())
        labels = labels.view(-1, 1)

        outputs = inputs @ model
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        model_optimizer.step()
        model_optimizer.zero_grad()
loss = criterion(x.view(-1, 1) @ model_clone, y.view(-1, 1))
model_clone_optimizer.zero_grad()
loss.backward()
model_clone_optimizer.step()
print(f"w/ accumulation, the final model weight is {model.mean().item():.5f}")
print(f"w/o accumulation, the final model weight is {model_clone.mean().item():.5f}")
