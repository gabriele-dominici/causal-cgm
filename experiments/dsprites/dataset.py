import os.path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm


# Helper function to convert grayscale image to red or green
def colorize(image, color):
    colored_image = torch.zeros(3, 64, 64)  # Create an image with 3 channels (RGB)
    if color == 'red':
        colored_image[0] = image  # Red channel
    elif color == 'green':
        colored_image[1] = image  # Green channel
    elif color == 'blue':
        colored_image[2] = image  # Green channel
    return colored_image


# Creating the custom dataset
class CustomDSpritesDataset(torch.utils.data.Dataset):
    def __init__(self, dsprites_dataset, concept_label, target_label):
        self.dsprites_dataset = dsprites_dataset
        self.concept_label = concept_label
        self.target_label = target_label

    def __len__(self):
        return len(self.dsprites_dataset)

    def __getitem__(self, idx):
        image = self.dsprites_dataset[idx]
        concept_label = self.concept_label[idx]
        target_label = self.target_label[idx]

        # Colorize the image
        if concept_label[4] == 1:
            color = 'green'
        else:
            color = 'red'
        colored_image = colorize(image.squeeze(), color)  # Remove channel dimension of the grayscale image

        return colored_image, torch.tensor(concept_label, dtype=torch.float32), torch.tensor(target_label, dtype=torch.float32)


def load_preprocessed_data(base_dir='./experiments/dsprites'):
    train_features = torch.from_numpy(np.load(os.path.join(base_dir, 'train_features.npy')))
    train_concepts = torch.from_numpy(np.load(os.path.join(base_dir, 'train_concepts.npy')))
    train_tasks = torch.from_numpy(np.load(os.path.join(base_dir, 'train_tasks.npy'))).unsqueeze(1)
    test_features = torch.from_numpy(np.load(os.path.join(base_dir, 'test_features.npy')))
    test_concepts = torch.from_numpy(np.load(os.path.join(base_dir, 'test_concepts.npy')))
    test_tasks = torch.from_numpy(np.load(os.path.join(base_dir, 'test_tasks.npy'))).unsqueeze(1)
    return train_features, train_concepts, train_tasks, test_features, test_concepts, test_tasks



def main():
    # Step 1: Prepare the MNIST dataset
    # [Include the CustomMNISTDataset class from the previous code snippet here]

    # Load dSprites dataset
    # load np arrays
    dsprites_train_img = torch.from_numpy(np.load('../../datasets/dsprites/train_images.npy'))
    dsprites_train_concepts = torch.from_numpy(np.load('../../datasets/dsprites/train_concepts.npy'))
    dsprites_train_labels = torch.from_numpy(np.load('../../datasets/dsprites/train_labels.npy'))
    dsprites_test_img = torch.from_numpy(np.load('../../datasets/dsprites/test_images.npy'))
    dsprites_test_concepts = torch.from_numpy(np.load('../../datasets/dsprites/test_concepts.npy'))
    dsprites_test_labels = torch.from_numpy(np.load('../../datasets/dsprites/test_labels.npy'))

    # Create custom datasets
    custom_train_dataset = CustomDSpritesDataset(dsprites_train_img, dsprites_train_concepts, dsprites_train_labels)
    custom_test_dataset = CustomDSpritesDataset(dsprites_test_img, dsprites_test_concepts, dsprites_test_labels)

    # DataLoaders
    train_loader = DataLoader(custom_train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(custom_test_dataset, batch_size=64, shuffle=False)

    # Step 2: Prepare ResNet18 model for feature extraction
    model = models.resnet18(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Step 3: Extract features
    def extract_features(data_loader):
        features = []
        concept_labels = []
        task_labels = []

        with torch.no_grad():
            for imgs, concepts, tasks in tqdm(data_loader):
                imgs = imgs.to(device)
                out = model(imgs)
                features.append(out.cpu().numpy())
                concept_labels.append(concepts.numpy())
                task_labels.append(tasks.numpy())

        return np.concatenate(features), np.concatenate(concept_labels), np.concatenate(task_labels)

    train_features, train_concepts, train_tasks = extract_features(train_loader)
    test_features, test_concepts, test_tasks = extract_features(test_loader)

    # Step 4: Save the embeddings and labels
    np.save('../../datasets/dsprites/train_features.npy', train_features)
    np.save('../../datasets/dsprites/train_concepts.npy', train_concepts)
    np.save('../../datasets/dsprites/train_tasks.npy', train_tasks)

    np.save('../../datasets/dsprites/test_features.npy', test_features)
    np.save('../../datasets/dsprites/test_concepts.npy', test_concepts)
    np.save('../../datasets/dsprites/test_tasks.npy', test_tasks)


if __name__ == "__main__":
    main()
