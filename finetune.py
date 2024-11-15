import torch
from torch import nn
import torchvision.transforms as T
from models import FlexibleGANForSegmentation
import argparse
import numpy as np
import random
import pandas as pd
import datetime
import torchmetrics as TM
from torch.utils.data import Subset

from finetune_methods import (
    OxfordIIITPetsAugmented,
    TrimapClasses,
    args_to_dict,
    tensor_trimap
)

def train_loop(model, trainloader, valloader, num_epochs, optimizer, scheduler, criterion, device, model_save_path, print_images=False):
    """
    Executes the training and validation loop for the given model using specified data loaders and training parameters.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader for the training dataset.
        valloader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Number of epochs to train the model.
        optimizer (Optimizer): Optimizer for adjusting model weights.
        scheduler (lr_scheduler): Scheduler for adjusting the learning rate.
        criterion (loss): Loss function to measure model performance.
        device (torch.device): Device (CPU/GPU) to perform training.
        model_save_path (str): Path to save the model after training.
        print_images (bool, optional): If True, print images during training (default: False).

    Returns:
        tuple: Contains the trained model and a tuple of lists with training and validation statistics 
               (train_losses, train_pixaccs, train_IoU, test_losses, test_pixaccs, test_IoU).
    """
    test_losses = []
    test_pixaccs = []
    test_IoU = []

    pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro')
    pixel_metric = pixel_metric.to(device)
    iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND)
    iou = iou.to(device)    

    train_losses = []
    train_pixaccs = []
    train_IoU = []

    best_loss = np.inf

    # Iterate over each epoch for training and evaluation
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}')
        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        running_samples = 0

        train_IoU_batch = []
        train_pixacc_batch = []
        train_loss_batch = []
        
        # Iterate through each batch in the training loader        
        for batch_idx, (inputs, targets) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            targets = targets.squeeze(dim=1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
            running_samples += targets.size(0)
            running_loss += loss.item()
            epoch_loss += loss.item()

            if (batch_idx+1)%50 == 0:
                print(f"Batch {batch_idx+1}/{len(trainloader)}, Loss: {running_loss /50:.4f}")
                running_loss = 0.0
            
            # Get test predictions
            pred = nn.Softmax(dim=1)(outputs)
            pred_labels = pred.argmax(dim=1)
            pred_mask = pred_labels.to(torch.float)

            # Compute metrics
            pixel_accuracy = pixel_metric(pred_labels, targets)
            iou_accuracy = iou(pred_mask, targets)

            # Move metrics tensors to cpu
            loss = loss.cpu()
            pixel_accuracy = pixel_accuracy.cpu()
            iou_accuracy = iou_accuracy.cpu()

            train_IoU_batch.append(iou_accuracy)
            train_pixacc_batch.append(pixel_accuracy)
            train_loss_batch.append(loss.item())

        train_losses.append(np.mean(train_loss_batch))
        train_pixaccs.append(np.mean(train_pixacc_batch))
        train_IoU.append(np.mean(train_IoU_batch))

        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss / len(trainloader):.4f}")
        
        #Run evaluation
        model.eval()
        with torch.no_grad():
            test_loss_batch = []
            test_pixacc_batch = []
            test_IoU_batch = []

            for (test_inputs, test_labels) in valloader:
                # Get test predictions
                test_inputs = test_inputs.to(device)
                test_outputs = model(test_inputs)
                test_labels = test_labels.squeeze(dim=1)
                test_labels = test_labels.to(device)
                test_pred = nn.Softmax(dim=1)(test_outputs)
                pred_labels = test_pred.argmax(dim=1)
                pred_mask = pred_labels.to(torch.float)

                # Compute metrics
                test_loss = criterion(test_outputs, test_labels)
                pixel_accuracy = pixel_metric(pred_labels, test_labels)
                iou_accuracy = iou(pred_mask, test_labels)

                # Move metrics tensors to cpu
                test_loss = test_loss.cpu()
                pixel_accuracy = pixel_accuracy.cpu()
                iou_accuracy = iou_accuracy.cpu()

                test_IoU_batch.append(iou_accuracy)
                test_pixacc_batch.append(pixel_accuracy)
                test_loss_batch.append(test_loss.item())

            test_losses.append(np.mean(test_loss_batch))
            test_pixaccs.append(np.mean(test_pixacc_batch))
            test_IoU.append(np.mean(test_IoU_batch))

            print(f"Validation Loss: {np.mean(test_loss_batch):.4f}, Pix Acc: {np.mean(test_pixacc_batch):.4f}, IoU: {np.mean(test_IoU_batch):.4f}\n")

            if np.mean(test_loss_batch) < best_loss:
                torch.save(model.state_dict(), model_save_path + ".pth")
                best_loss = np.mean(test_loss_batch)
            
        if scheduler is not None:
            scheduler.step()
        
    return model, (train_losses, train_pixaccs, train_IoU, test_losses, test_pixaccs, test_IoU)


def train(model_path = None, percent=1.0, num_epochs=20):
    """
    Trains a model on the Oxford IIIT Pet dataset with optional fine-tuning from a pre-trained model.

    Args:
        model_path (str, optional): Path to a pre-trained model for fine-tuning.
        percent (float): Percentage of the training data to use, between 0.1 and 1.0.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        str: Path to the saved statistics of the training process.
    """
    seed = 42
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')
    
    if percent < 0.1:
        raise ValueError(f'Percentage of data too low, should be at least 0.1, value given: {percent}')
    if percent > 1.0:
        raise ValueError(f'Percentage of data too high, should be maximum 1.0, value given: {percent}')
    
    # Define hyperparameters
    batch_size = 16
    learning_rate = 0.0004
    image_size = 64

    print_images = False
    data_folder = './data/fine-tuning'
    output_model_folder = './output/models/'
    output_stats_folder = './output/stats/'

    # Load the model, optionally with a pre-trained base
    model = FlexibleGANForSegmentation(output_size=image_size, model_filepath=model_path)
    model.to(device)

    # Define transformations for data preprocessing
    transform_dict = args_to_dict(
        pre_transform=T.ToTensor(),
        pre_target_transform=T.ToTensor(),
        common_transform=T.Compose([T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),]),
        post_target_transform=T.Compose([T.Lambda(tensor_trimap),]),
        )
    # Prepare datasets
    pets_trainval = OxfordIIITPetsAugmented(
        root=data_folder,
        split="trainval",
        target_types="segmentation",
        download=False,
        **transform_dict,
        )
    
    # Split dataset into train and validation
    indices = np.arange(len(pets_trainval))
    np.random.shuffle(indices)
    train_indices = indices[:int(len(pets_trainval)*0.8)]
    val_indices = indices[int(len(pets_trainval)*0.8):]
    pets_train = Subset(pets_trainval, train_indices)
    pets_val = Subset(pets_trainval, val_indices)

    # Select percentage subset of training set
    if percent != 1.0:
        indices = np.arange(len(pets_train))
        np.random.shuffle(indices)
        subset_indices = indices[:int(len(pets_train)*percent)]
        pets_train = Subset(pets_train, subset_indices)

    # pets_test = OxfordIIITPetsAugmented(
    #     root=data_folder,
    #     split="test",
    #     target_types="segmentation",
    #     download=False,
    #     **transform_dict,
    #     )
    
    pets_train_loader = torch.utils.data.DataLoader(
        pets_train,
        batch_size=batch_size,
        shuffle=True,
        )
    pets_val_loader = torch.utils.data.DataLoader(
        pets_val,
        batch_size=batch_size,
        shuffle=True,
        )
    # pets_test_loader = torch.utils.data.DataLoader(
    #     pets_test,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    #criterion = IoULoss(softmax=True)

    # Set model name based on the pre-training configuration and percent of data used
    if model_path is not None:
        if "jigsaw4" in model_path or 'j4' in model_path:
            model_name_out = 'ft_j4_' + str(int(percent*100)) + 'p'
        elif "jigsaw9" in model_path or 'j9' in model_path:
            model_name_out = 'ft_j9_' + str(int(percent*100)) + 'p'
        else:
            model_name_out = 'ft_j0_' + str(int(percent*100)) + 'p'
    else:
        model_name_out = 'b_' + str(int(percent*100)) + 'p'

    model, val_calcs = train_loop(
        model=model, 
        trainloader=pets_train_loader, 
        valloader=pets_val_loader, 
        num_epochs=num_epochs, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        criterion=criterion, 
        device=device,
        print_images=print_images,
        model_save_path=output_model_folder+model_name_out
        )
    
    stats = {
        'train_loss': val_calcs[0],
        'train_accuracy': val_calcs[1],
        'train_iou': val_calcs[2],
        'val_loss': val_calcs[3],
        'val_accuracy': val_calcs[4],
        'val_iou': val_calcs[5]
    }

    stats = pd.DataFrame(stats)
    
    model_date = f'{datetime.datetime.now():%Y%m%d_%H%m}_'

    # torch.save(model.state_dict(), output_model_folder + model_name_out + ".pth")
    stats_path = output_stats_folder + 'stats_' + model_name_out + '.csv'
    stats.to_csv(stats_path)
    
    return stats_path


def test(model_path=None):
    """
    Evaluates a pre-trained model on the test dataset and reports the loss, pixel accuracy and IoU.

    Args:
        model_path (str, optional): Path to the pre-trained model file.

    Returns:
        tuple: A tuple containing mean loss, pixel accuracy, and IoU scores across the test dataset.
    """
    seed = 42
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')

    batch_size = 16
    learning_rate = 0.0004
    image_size = 64

    # Load the model and set up the transformation pipeline
    model = FlexibleGANForSegmentation(output_size=image_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)

    transform_dict = args_to_dict(
        pre_transform=T.ToTensor(),
        pre_target_transform=T.ToTensor(),
        common_transform=T.Compose([T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),]),
        post_target_transform=T.Compose([T.Lambda(tensor_trimap),]),
        )
    
    data_folder = './data/fine-tuning'
    
    pets_test = OxfordIIITPetsAugmented(
        root=data_folder,
        split="test",
        target_types="segmentation",
        download=False,
        **transform_dict,
        )

    pets_test_loader = torch.utils.data.DataLoader(
        pets_test,
        batch_size=21,
        shuffle=True,
    )

    # Initialize metrics for accuracy and IoU
    pixel_metric = TM.classification.MulticlassAccuracy(3, average='micro')
    pixel_metric = pixel_metric.to(device)
    iou = TM.classification.MulticlassJaccardIndex(3, average='micro', ignore_index=TrimapClasses.BACKGROUND)
    iou = iou.to(device) 
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_loss_list = []
        test_pixacc_list = []
        test_IoU_list = []

        for (test_inputs, test_labels) in pets_test_loader:
            # Get test predictions
            test_inputs = test_inputs.to(device)
            test_outputs = model(test_inputs)
            test_labels = test_labels.squeeze(dim=1)
            test_labels = test_labels.to(device)
            test_pred = nn.Softmax(dim=1)(test_outputs)
            pred_labels = test_pred.argmax(dim=1)
            pred_mask = pred_labels.to(torch.float)

            # Compute metrics
            test_loss = criterion(test_outputs, test_labels)
            pixel_accuracy = pixel_metric(pred_labels, test_labels)
            iou_accuracy = iou(pred_mask, test_labels)

            # Move metrics tensors to cpu
            test_loss = test_loss.cpu()
            pixel_accuracy = pixel_accuracy.cpu()
            iou_accuracy = iou_accuracy.cpu()

            test_IoU_list.append(iou_accuracy)
            test_pixacc_list.append(pixel_accuracy)
            test_loss_list.append(test_loss.item())

    # Calculate and return the mean values of the loss, pixel accuracy, and IoU
    loss = np.mean(test_loss_list)
    pixacc = np.mean(test_pixacc_list)
    iou_calc = np.mean(test_IoU_list)

    return((loss, pixacc, iou_calc))


