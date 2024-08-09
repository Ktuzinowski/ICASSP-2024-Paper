from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import time
import torch
# Constants
img_size = 224
train_batch_size = 128
eval_batch_size = 64

transform_train = transforms.Compose([
    transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
print("Got through the setting up of the transforms")
trainset = datasets.ImageFolder('imagenet/train', transform=transform_train)
testset = datasets.ImageFolder('imagenet/val', transform=transform_test)
print("Got through getting the Test and Validation Set for the Data we are Loading in")
train_sampler = RandomSampler(trainset)
test_sampler = SequentialSampler(testset)
print("Got through the samplers")
train_loader = DataLoader(trainset,
                          sampler=train_sampler,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True)
test_loader = DataLoader(testset,
                         sampler=test_sampler,
                         batch_size=eval_batch_size,
                         num_workers=4,
                         pin_memory=True) if testset is not None else None
print("Got the Data Loaders to Work")
# Function to check a few batches from the dataloader
def check_dataloader(loader, num_batches=2):
    start_time = time.time()
    for i, (inputs, labels) in enumerate(loader):
        if i >= num_batches:
            break
        print(f"Batch {i + 1}")
        print(f"Inputs shape: {inputs.shape}")
        print(f"Labels shape: {labels.shape}")
        if loader.pin_memory and inputs.is_pinned():
            print("Data is pinned")
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            print("Data moved to GPU")
    end_time = time.time()
    print(f"Time to load {num_batches} batches: {end_time - start_time:.2f} seconds")

# Check the train loader
print("Checking train_loader:")
check_dataloader(train_loader)

# Check the test loader
if test_loader:
    print("\nChecking test_loader:")
    check_dataloader(test_loader)
