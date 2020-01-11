import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import warnings

warnings.filterwarnings('ignore')

DEBUG = False
BATCH_SIZE = 64
NUM_CLASSES = 39

data_dir = './images'


def load_split_train_test(datadir, valid_size=.35):
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(256),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    #  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)

    num_train = len(train_data)
    print('train images size is:{}'.format(num_train))

    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=BATCH_SIZE)

    return trainloader, testloader


trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
print(len(trainloader.dataset.classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel + 2 * padding) / stride) + 1
    return output


class SimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 25, 5)

        self.fc1 = nn.Linear(25 * 12 * 12, 1200)
        self.fc2 = nn.Linear(1200, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, NUM_CLASSES)

    def forward(self, x):
        if DEBUG:
            print('first_shape {}'.format(x.shape))
        x = self.pool(F.relu(self.conv1(x)))
        if DEBUG:
            print('second_shape {}'.format(x.shape))
        x = self.pool(F.relu(self.conv2(x)))
        if DEBUG:
            print('4nd_shape{}'.format(x.shape))
        x = self.pool(F.relu(self.conv3(x)))
        if DEBUG:
            print('5nd_shape{}'.format(x.shape))
        x = self.pool(F.relu(self.conv4(x)))
        if DEBUG:
            print('6nd shape efore reshape{}'.format(x.shape))
        x = x.view(-1, 25 * 12 * 12)
        if DEBUG:
            print('7nd_shape after reshape{}'.format(x.shape))
        x = F.relu(self.fc1(x))
        if DEBUG:
            print('8nd_shape {}'.format(x.shape))
        x = F.relu(self.fc2(x))
        if DEBUG:
            print('9nd_shape {}'.format(x.shape))
        x = F.relu(self.fc3(x))
        if DEBUG:
            print('10nd_shape {}'.format(x.shape))
        x = self.fc4(x)
        if DEBUG:
            print('11nd_shape {}'.format(x.shape))
        return x


cnn = SimpleCNN()
cnn = cnn.to(device)

print(len(trainloader.dataset.classes))


def createLossAndOptimizer(net, learning_rate=0.001):
    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)


def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train_network(tr_loader, criterion, optim,
                  device='cuda', net=SimpleCNN(), n_epoch=5):
    net = net.to(device)
    for epoch in range(n_epoch):
        running_loss = 0.0
        for i, data in enumerate(tr_loader, 0):
            # getting inputs and labels for batch
            inputs, labels = data[0].to(device), data[1].to(device)
            optim.zero_grad()
            # forward pass
            outputs = net.forward(inputs)
            _, out = torch.max(outputs.data, 1)
            if DEBUG:
                print('outputs shape {}'.format(outputs.shape))
                print('labels shape {}'.format(labels.shape))
                print(outputs[0])
                raise Exception()
            loss = criterion(outputs, labels)
            # backward pass
            loss.backward()
            optim.step()
            # print what we've got
            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0


loss, optimizer = createLossAndOptimizer(cnn)
train_network(trainloader, loss, optimizer, device, cnn, 35)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        output = cnn(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network is %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(NUM_CLASSES))
class_total = list(0. for i in range(NUM_CLASSES))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(NUM_CLASSES):
    print('Accuracy of %10s : %2d %%' % (
        trainloader.dataset.classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(cnn.state_dict(), 'last_cnn_dict.pt')
torch.save(cnn, 'last_cnn.pt')
example = torch.rand(1, 3, 256, 256)
ex = example.to(device)
traced_script_module = torch.jit.trace(cnn, ex)
traced_script_module.save('last_jit_model.pt')
