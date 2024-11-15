import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
import torchmetrics as TM
from enum import IntEnum


class OxfordIIITPetsAugmented(torchvision.datasets.OxfordIIITPet):
    """
    Enhances the OxfordIIITPet dataset class to allow for more complex preprocessing and transformations.
    
    Args:
        root (str): Directory where the dataset is located or will be downloaded.
        split (str): Indicates the dataset split to use, typically 'trainval' or 'test'.
        target_types (str): Specifies the type of target data ('segmentation', 'classification', etc.).
        download (bool): If True, downloads the dataset from the internet if it's not already available at the root path.
        pre_transform (callable, optional): Transformation applied to the images before any split specific processing.
        post_transform (callable, optional): Transformation applied to the images after splitting into train/validate.
        pre_target_transform (callable, optional): Transformation applied to the labels before any split specific processing.
        post_target_transform (callable, optional): Transformation applied to the labels after splitting into train/validate.
        common_transform (callable, optional): Transformation applied to both images and labels together before splitting.
        
    Attributes:
        post_transform (callable): Stores the post-processing transformation for images.
        post_target_transform (callable): Stores the post-processing transformation for labels.
        common_transform (callable): Stores the common transformation applied to both images and labels.
    """    
    def __init__(
        self,
        root: str,
        split: str,
        target_types="segmentation",
        download=False,
        pre_transform=None,
        post_transform=None,
        pre_target_transform=None,
        post_target_transform=None,
        common_transform=None,
    ):
        super().__init__(
            root=root,
            split=split,
            target_types=target_types,
            download=download,
            transform=pre_transform,
            target_transform=pre_target_transform,
        )
        self.post_transform = post_transform
        self.post_target_transform = post_target_transform
        self.common_transform = common_transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.        
        Returns:
            int: Total number of samples.
        """
        return super().__len__()

    def __getitem__(self, idx):
        """
        Retrieves an item by index from the dataset after applying transformations.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: Contains the transformed image and label.
        """
        (input, target) = super().__getitem__(idx)
        if self.common_transform is not None:
            both = torch.cat([input, target], dim=0)
            both = self.common_transform(both)
            (input, target) = torch.split(both, 3, dim=0)
        
        if self.post_transform is not None:
            input = self.post_transform(input)
        if self.post_target_transform is not None:
            target = self.post_target_transform(target)

        return (input, target)
    
class IoULoss(nn.Module):
    """
    Custom loss function to compute Intersection over Union (IoU) loss.
    Args:
        softmax (bool): If True, applies a softmax function to predictions before computing IoU.
    """
    def __init__(self, softmax=False):
        super().__init__()
        self.softmax = softmax
    
    def forward(self, pred, gt):
        """
        Calculates the negative log of IoU metric as the loss.
        Args:
            pred (torch.Tensor): Predicted outputs from the model.
            gt (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: Computed IoU loss.
        """
        return -(IoUMetric(pred, gt, self.softmax).log())


class TrimapClasses(IntEnum):
    PET = 0
    BACKGROUND = 1
    BORDER = 2


def save_model_checkpoint(model, cp_name):
    """
    Saves the model's state dictionary to a file.
    Args:
        model (nn.Module): The model to save.
        cp_name (str): Path and filename where the checkpoint will be saved.
    """
    torch.save(model.state_dict(), cp_name)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model_from_checkpoint(model, ckp_path):
    return model.load_state_dict(
        torch.load(
            ckp_path,
            map_location=get_device(),
        )
    )


def get_model_parameters(m):
    """
    Calculates the total number of trainable parameters in a model.
    Args:
        m (nn.Module): The model.
    Returns:
        int: Total number of trainable parameters.
    """
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params


def close_figures():
    """
    Closes all open matplotlib figures.
    """
    while len(plt.get_fignums()) > 0:
        plt.close()

def trimap2f(trimap):
    """
    Converts a trimap tensor to a format usable for visualization or further processing.
    Args:
        trimap (torch.Tensor): Trimap tensor.
    Returns:
        PIL.Image: Processed trimap converted to a PIL Image.
    """
    return (T.ToPILImage(trimap) * 255.0 - 1) / 2

    
def tensor_trimap(t):
    """
    Converts a tensor to a trimap format by scaling and adjusting values.
    """
    x = t * 255
    x = x.to(torch.long)
    x = x - 1
    return x

def args_to_dict(**kwargs):
    return kwargs

def IoUMetric(pred, gt, softmax=False):
    """
    Computes the Intersection over Union (IoU) metric.
    Args:
        pred (torch.Tensor): Predictions from the model.
        gt (torch.Tensor): Ground truth labels.
        softmax (bool): If True, applies softmax to the predictions.
    Returns:
        torch.Tensor: The mean IoU across all samples.
    """
    if softmax is True:
        pred = nn.Softmax(dim=1)(pred)
    
    gt = torch.cat([ (gt == i) for i in range(3) ], dim=1)
    intersection = gt * pred
    union = gt + pred - intersection
    iou = (intersection.sum(dim=(1, 2, 3)) + 0.001) / (union.sum(dim=(1, 2, 3)) + 0.001)

    return iou.mean()

def test_custom_iou_loss():
    """
    Test function to validate the IoU loss computation.
    Returns:
        torch.Tensor: The computed IoU loss for a sample input and label.
    """
    x = torch.rand((2, 3, 2, 2), requires_grad=True)
    y = torch.randint(0, 3, (2, 1, 2, 2), dtype=torch.long)
    z = IoULoss(softmax=True)(x, y)

    return z

def test_dataset_accuracy(model, loader, device):
    """
    Evaluates model accuracy on a dataset using IoU and pixel accuracy metrics.
    Args:
        model (nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform the evaluation on.
    """
    model.eval()
    iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND)
    iou = iou.to(device)
    pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro')
    pixel_metric = pixel_metric.to(device)
    
    iou_accuracies = []
    pixel_accuracies = []
    custom_iou_accuracies = []

    # The loop iterates over each batch of data provided by the DataLoader
    for batch_idx, (inputs, targets) in enumerate(loader, 0):
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = inputs.to(device)
        
        pred_probabilities = nn.Softmax(dim=1)(predictions)
        pred_labels = predictions.argmax(dim=1)

        # Add a value 1 dimension at dim=1
        pred_labels = pred_labels.unsqueeze(1)
        # print("pred_labels.shape: {}".format(pred_labels.shape))
        pred_mask = pred_labels.to(torch.float)

        iou_accuracy = iou(pred_mask, targets)
        # pixel_accuracy = pixel_metric(pred_mask, targets)
        pixel_accuracy = pixel_metric(pred_labels, targets)
        custom_iou = IoUMetric(pred_probabilities, targets)
        iou_accuracies.append(iou_accuracy.item())
        pixel_accuracies.append(pixel_accuracy.item())
        custom_iou_accuracies.append(custom_iou.item())

    
    iou_tensor = torch.FloatTensor(iou_accuracies)
    pixel_tensor = torch.FloatTensor(pixel_accuracies)
    custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)
    
    print("Test Dataset Accuracy")
    print(f"Pixel Accuracy: {pixel_tensor.mean():.4f}, IoU Accuracy: {iou_tensor.mean():.4f}, Custom IoU Accuracy: {custom_iou_tensor.mean():.4f}")


def prediction_accuracy(ground_truth_labels, predicted_labels):
    """
    Calculates the accuracy of predictions by comparing predicted labels to ground truth labels.

    Args:
        ground_truth_labels (torch.Tensor): The actual labels of the data as provided by the dataset.
        predicted_labels (torch.Tensor): The labels predicted by the model.

    Returns:
        float: The accuracy of the predictions expressed as the ratio of correct predictions to the total number of predictions made. 
        This is calculated by comparing each element of the predicted labels with the ground truth, summing the true comparisons, and then dividing by the total number of labels.
    """
    eq = ground_truth_labels == predicted_labels
    return eq.sum().item() / predicted_labels.numel()


def print_test_dataset_masks(model, test_pets_targets, test_pets_labels, epoch, show_plot, device):
    """
    Evaluates and displays model predictions alongside the ground truth masks for visual comparison.
    
    Args:
        model (nn.Module): The trained model used for prediction.
        test_pets_targets (torch.Tensor): The input images for which predictions are to be made.
        test_pets_labels (torch.Tensor): The ground truth labels for the input images.
        epoch (int): Current epoch number, used to label the output images for reference.
        show_plot (bool): If True, display the plots; if False, close the figure without displaying.
        device (torch.device): The device (CPU or GPU) on which the computation is performed.
    """
    model.eval()
    predictions = model(test_pets_targets)
    test_pets_labels = test_pets_labels.to(device)
    pred = nn.Softmax(dim=1)(predictions)

    pred_labels = pred.argmax(dim=1)
    pred_labels = pred_labels.unsqueeze(1)
    pred_mask = pred_labels.to(torch.float)

    iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND)
    iou = iou.to(device)
    iou_accuracy = iou(pred_mask, test_pets_labels)
    pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro')
    pixel_metric = pixel_metric.to(device)
    pixel_accuracy = pixel_metric(pred_labels, test_pets_labels)
    custom_iou = IoUMetric(pred, test_pets_labels)
    title = f'Epoch: {epoch:02d}, Accuracy[Pixel: {pixel_accuracy:.4f}, IoU: {iou_accuracy:.4f}, Custom IoU: {custom_iou:.4f}]'
    print(title)

    close_figures()
    
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=12)

    fig.add_subplot(3, 1, 1)
    plt.imshow(T.ToPILImage(torchvision.utils.make_grid(test_pets_targets, nrow=7)))
    plt.axis('off')
    plt.title("Targets")

    fig.add_subplot(3, 1, 2)
    plt.imshow(T.ToPILImage(torchvision.utils.make_grid(test_pets_labels.float() / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Ground Truth Labels")

    fig.add_subplot(3, 1, 3)
    plt.imshow(T.ToPILImage(torchvision.utils.make_grid(pred_mask / 2.0, nrow=7)))
    plt.axis('off')
    plt.title("Predicted Labels")
    
    if show_plot is False:
        close_figures()
    else:
        plt.show()

