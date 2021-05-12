
""" from skimage import io, color
rgb = io.imread(filename)
lab = color.rgb2lab(rgb)
 """

'''
# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# https://medium.com/@sergioalves94/deep-learning-in-pytorch-with-cifar-10-dataset-858b504a6b54
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    trainset = torchvision.datasets.CIFAR10(
       root='./data', train=True, download=True, transform=transform_train)
   trainloader = torch.utils.data.DataLoader(
       trainset, batch_size=128, shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR10(
       root='./data', train=False, download=True, transform=transform_test)
   testloader = torch.utils.data.DataLoader(
       testset, batch_size=100, shuffle=False, num_workers=2)
    '''
