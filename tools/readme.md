# Tools for ML developments

<details>
<summary><b>Table of Contents</b> (click to open)</summary>

- [Linux](#linux)
- [Python](#python)
  - [Typing hints](#typing-hints)
  - [Pytest](#pytest)
  - [Logging with rich](#logging-with-rich)
  - [Using Hydra to configure applications](#using-hydra-to-configure-applications)
- [Kubernetes](#kubernetes)
- [PyTorch](#pytorch)
    - [PyTorch hooks](#pytorch-hooks)
    - [Out of memory](#out-of-memory)
    - [Training with tensorboardX](#training-with-tensorboardx)
    - [Pytorch Profiler](#pytorch-profiler)

</details>

## Linux
```bash
tar -zcvf examples.tgz examples # package a examples.tgz
tar -zxvf examples.tgz # unzip examples.tgz
lsof -n -i :xxx | grep LISTEN # check pid that using xxx port
```

## Python

### Typing hints
```python
from typing import Dict, List, Union, Optional

def add(a: int, b: int) -> Optional(int):
	if a < 10:
		return a+b
	else:
		return None

def sort_list(a: List[int]) -> List[int]:
	# do sort
	return a

def parse_config(config_dir: str) -> Dict[str, str]:
	# parse params
	return {'xxx': 'xxx'}

def response(url: str, data: Dict[str, str]) -> Union[int, str]:
	res = requests.post(url, data=data)
	if res.status_code == 200:
		return 200
	else:
		return res.text
```
### Pytest
```python
import pytest

def add(a: int, b: int):
	return a + b

def test_answer():
	assert add(1, 2) == 3

@pytest.mark.parametrize("test_input,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)])
def test_eval(test_input, expected):
    assert eval(test_input) == expected
```

### Logging with [rich](https://github.com/Textualize/rich)
```python
# 1. print with color
from rich.console import Console

console = Console()

console.print("Hello", "World!", style="bold red") 

# 2. helper function
from rich import inspect

my_list = ["foo", "bar"]
inspect(my_list, methods=True) 

# 3. progress bar
from rich.progress import track

for step in track(range(100)):
    do_step(step) 

# 4. table
from rich.console import Console
from rich.table import Table

console = Console()

table = Table(show_header=True, header_style="bold magenta")
table.add_column("Date", style="dim", width=12)
table.add_column("Title")
table.add_column("Production Budget", justify="right")
table.add_column("Box Office", justify="right")
table.add_row(
    "Dec 20, 2019", "Star Wars: The Rise of Skywalker", "$275,000,000", "$375,126,118"
)
table.add_row(
    "May 25, 2018",
    "[red]Solo[/red]: A Star Wars Story",
    "$275,000,000",
    "$393,151,347",
)
table.add_row(
    "Dec 15, 2017",
    "Star Wars Ep. VIII: The Last Jedi",
    "$262,000,000",
    "[bold]$1,332,539,889[/bold]",
)

console.print(table)

# status
from time import sleep
from rich.console import Console

console = Console()
tasks = [f"task {n}" for n in range(1, 11)]

with console.status("[bold green]Working on tasks...") as status:
    while tasks:
        task = tasks.pop(0)
        sleep(1)
        console.log(f"{task} complete")
```

### Using Hydra to configure applications

```bash
# conf/config.yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

```python
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

You can run function multiple times with different configs.

```bash
├── conf
│   ├── config.yaml
│   ├── db
│   │   ├── mysql.yaml
│   │   └── postgresql.yaml
│   └── __init__.py
└── my_app.py
```

```bash
python my_app.py --multirun db=mysql,postgresql
```

## Kubernetes

## PyTorch

### PyTorch hooks
```python
import torch
import torch.nn as nn


# Define a simple neural network for demonstration purposes
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# Create an instance of the neural network
net = SimpleNet()


# Define a forward hook function
def forward_hook(module, input, output):
    print(f"Forward Hook - Input shape: {input[0].shape}, Output shape: {output.shape}")


# Register the forward hook on the first fully connected layer (fc1)
handle_fc1_forward = net.fc1.register_forward_hook(forward_hook)


# Define a backward hook function
def backward_hook(module, grad_input, grad_output):
    print(f"Backward Hook - Gradient of input: {grad_input[0].shape}, Gradient of output: {grad_output[0].shape}")


# Register the backward hook on the second fully connected layer (fc2)
handle_fc2_backward = net.fc2.register_backward_hook(backward_hook)


# Create some random input data
input_data = torch.randn(1, 10)

# Forward pass through the network
output = net(input_data)

# Calculate the loss (for demonstration, we'll just use a simple mean squared error loss)
loss = torch.mean(output ** 2)

# Backward pass to compute gradients
loss.backward()

# Remove the hooks when you're done with them
handle_fc1_forward.remove()
handle_fc2_backward.remove()
```

### Out of memory

### Training with tensorboardX
```python
import torch
import tensorboardX

writer = tensorboardX.SummaryWriter('path-to-log')

loss_val = "xxx"
lr = "xxx"
global_step = "xxx"

writer.add_scalar("train/loss", loss_val, global_step)
writer.add_scalar("train/lr", lr, global_step)

writer.close()
```
Check logs using `tensorboard --logdir 'path-to-log' --port 6001`.

### [Pytorch profiler](https://github.com/Jason-cs18/deep-learning-profiler)