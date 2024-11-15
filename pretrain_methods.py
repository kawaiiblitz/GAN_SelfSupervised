import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.datasets as datasets
import itertools
import random

class CustomImageDataset(Dataset):
    """
    A custom dataset class that loads images from subdirectories (each representing a class) within a specified directory
    and applies transformations if provided. This also assumes that each subdirectory's name is the label of the class.

    Args:
        directory (str): The root directory path where class subdirectories with images are stored.
        transform (callable, optional): A function that takes in a PIL image and returns a transformed version (default is None).
    """
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.classes = os.listdir(directory)  # List all directories/classes in the root directory
        self.classes.sort()  # Optional: sort class labels
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}  # Map class names to indices

        self.image_paths = []
        self.labels = []

        # Load all images, storing their paths and labels
        for cls in self.classes:
            cls_dir = os.path.join(directory, cls)  # Path to the class directory
            for img_filename in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_filename))
                self.labels.append(self.class_to_idx[cls])  # Store class index

    def __len__(self):
        """
        Return the number of images in the dataset.
        """    
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches the image at index `idx` from dataset, applies a transformation if specified, and returns it along with the label.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (transformed image, class label)
        """    
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply the transformation, if any

        return image



def shuffle_image(images, num_pieces, permutations, seed=None):
    """
    Shuffles a batch of images according to provided permutations, effectively creating a jigsaw puzzle effect.

    Args:
        images (torch.Tensor): A batch of images with shape [N, C, H, W], where N is the batch size, C is the number of channels, H is the height, and W is the width.
        num_pieces (int): The number of pieces to divide each image into. It must be a perfect square (e.g., 4, 9, 16) to form a square grid.
        permutations (list of lists): A list containing different permutations that dictate how to shuffle the image pieces.
        seed (int, optional): Optional random seed for reproducibility of shuffling.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The batch of shuffled images of shape [N, C, H, W].
            - torch.Tensor: The indices of the permutations used for each image in the batch.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    if len(images.shape) == 3:
        N = 1
        C, H, W = images.shape
    else:
        N, C, H, W = images.shape
    num_pieces_side = int(np.sqrt(num_pieces))
    piece_size = H // num_pieces_side
    batch_jigsaw = torch.zeros_like(images)    # Tensor to hold shuffled images
    perm_indices = []                          # List to hold the indices of the permutations used

    for n in range(N):
        image = images[n]
        perm_index = np.random.randint(len(permutations))
        permutation = permutations[perm_index]
        perm_indices.append(perm_index)

        # Rearrange image pieces according to the selected permutation
        for idx, piece_idx in enumerate(permutation):
            original_row = piece_idx // num_pieces_side
            original_col = piece_idx % num_pieces_side
            new_row = idx // num_pieces_side
            new_col = idx % num_pieces_side

            # Place the piece in its new position
            batch_jigsaw[n, :, new_row*piece_size:(new_row+1)*piece_size, new_col*piece_size:(new_col+1)*piece_size] = \
                image[:, original_row*piece_size:(original_row+1)*piece_size, original_col*piece_size:(original_col+1)*piece_size]

    return batch_jigsaw, torch.tensor(perm_indices)


def unshuffle_image(images, shuffled_indices, num_pieces, permutations):
    """
    Reverses the shuffling of a batch of images based on the provided permutations and indices,
    effectively solving the jigsaw puzzle to return the images to their original order.

    Args:
        images (torch.Tensor): A batch of shuffled images with shape [N, C, H, W], where
                               N is the batch size, C is the number of channels, H is the height, and W is the width.
        shuffled_indices (torch.Tensor): A tensor containing the indices of the permutations used to shuffle each image.
        num_pieces (int): The number of pieces each image was divided into.
        permutations (list of lists): A list containing different permutations used to shuffle the images.

    Returns:
        torch.Tensor: The batch of unshuffled (original) images of shape [N, C, H, W].
    """
    if len(images.shape) == 3:    # Check if there's only one image
        N = 1
        C, H, W = images.shape
    else:  # Multiple images
        N, C, H, W = images.shape
    # Calculate the number of pieces per side and the size of each piece
    num_pieces_side = int(np.sqrt(num_pieces))
    piece_size = H // num_pieces_side
    
    # Initialize an empty tensor for the original (unshuffled) image
    pred_images = torch.zeros_like(images)
    for n in range(N):
        image = images[n]
        permutation = permutations[shuffled_indices[n]]
    # Reverse the process of shuffling
        for new_idx, original_idx in enumerate(permutation):
            # Determine the position of the piece in the shuffled image
            new_row = new_idx // num_pieces_side
            new_col = new_idx % num_pieces_side
            
            # Determine the position of the piece in the original image
            original_row = original_idx // num_pieces_side
            original_col = original_idx % num_pieces_side
            
            # Place the piece in its original position
            pred_images[n, :, original_row*piece_size:(original_row+1)*piece_size, original_col*piece_size:(original_col+1)*piece_size] = \
            image[:, new_row*piece_size:(new_row+1)*piece_size, new_col*piece_size:(new_col+1)*piece_size]
    
    return pred_images


# Hamming distance function
def hamming_distance(p1, p2):
    """
    Calculate the Hamming distance between two sequences. It is the number of positions at which the corresponding elements are different.

    Args:
        p1 (iterable): First sequence.
        p2 (iterable): Second sequence.

    Returns:
        int: Hamming distance between the sequences.
    """
    return sum(el1 != el2 for el1, el2 in zip(p1, p2))


# Generation of the maximal Hamming distance permutation set
def generate_maximal_hamming_set(N, seed=None, printer=False):
    """
    Generates a set of permutations where successive permutations have the maximal cumulative Hamming distance
    from all previously selected permutations. This is typically used to ensure high diversity in selections.

    Args:
        N (int): Number of permutations to generate.
        seed (int, optional): Random seed for reproducibility.
        printer (bool, optional): If True, prints progress during the generation process.

    Returns:
        list of tuple: A list of permutations that maximizes the Hamming distance criterion.
    """
    if seed is not None:
        random.seed(seed)
    # All permutations of the set {1, ..., 9}
    all_perms = list(itertools.permutations(range(0, 9)))
    # Initialize P
    P = []
    # Repeat N times
    for i in range(N):
        # If first iteration, choose a random permutation to start
        if i == 0:
            j = random.randint(0, len(all_perms) - 1)
            P.append(all_perms.pop(j))
        else:
            # Calculate Hamming distances for all permutations in all_perms with respect to each in P
            D = np.array([[hamming_distance(p, perm) for perm in all_perms] for p in P])
            # Sum of Hamming distances for each permutation in all_perms
            D_bar = D.sum(axis=0)
            # Find permutation with maximal Hamming distance sum
            j = np.argmax(D_bar)
            # Add the best permutation to P
            P.append(all_perms.pop(j))
        if printer:
            print(f'{len(P)}/{N}')
    return P


def get_perms(num_pieces, num_permutations=30, seed=None, printer=False):
    """
    Get a list of permutations based on the number of pieces. Optionally uses a maximal Hamming distance algorithm to ensure diversity among permutations.

    Args:
        num_pieces (int): Number of pieces, determines the complexity of the permutation. Experiments are 4 or 9.
        num_permutations (int, optional): Number of permutations to generate if using maximal Hamming distance.
        seed (int, optional): Random seed for reproducibility.
        printer (bool, optional): If True, prints generation progress.

    Returns:
        list of tuple: A list of selected permutations.
    
    Raises:
        ValueError: If the number of pieces is not 4 or 9.
    """
    if num_pieces == 4:
        selected_perms = list(itertools.permutations(range(num_pieces)))   # Simple case, all permutations of 4
    elif num_pieces == 9:  # Complex case, generate permutations using maximal Hamming distance criterion
        selected_perms = generate_maximal_hamming_set(num_permutations, seed=seed, printer=printer)
    else:
        raise ValueError('Invalid number of pieces, should be 4 or 9.')
    return selected_perms
