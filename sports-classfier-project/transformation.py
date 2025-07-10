from torchvision import transforms

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __call__(self, img):
        return self.transform(img)
