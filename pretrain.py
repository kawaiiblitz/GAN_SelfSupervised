import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import itertools
from models import FlexibleGenerator, FlexibleDiscriminator
from pretrain_methods import CustomImageDataset, shuffle_image, get_perms
from torchmetrics import AUROC, F1Score, Accuracy
import pandas as pd
import argparse
import time
import os
import shutil
import random
import datetime

def train(jigsaw=True, num_pieces=9):
    """
    Initializes and runs the training process for a GAN model with optional jigsaw puzzle augmentation.
    
    Args:
        jigsaw (bool): Whether to apply jigsaw puzzle augmentation.
        num_pieces (int): Number of pieces to split an image into for jigsaw puzzle augmentation.

    Returns:
        str: Path to the saved discriminator model.
    """
    # Set random seed for reproducibility
    seed=42
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

    # Define hyperparameters
    num_epochs = 200
    batch_size = 16
    learning_rate = 0.0002
    image_size = 64
    latent_dim = 100
    print_batch = 50

    save_data = True

    if jigsaw is False:
        num_pieces = 0

    data_folder = 'data/pre-training/animals-10'
    output_folder = 'output_folder'
    
    print(f'Jigsaw Shuffling: {jigsaw}')
    
    param_str = f'image size: {image_size}, '
    if jigsaw:
        param_str += f'num_pieces: {num_pieces}, '
    param_str += f'num_epochs: {num_epochs}, latent_dim: {latent_dim}'
    
    print('Parameters selected ' + param_str)

    data_path = './data/pre-training/archive/animals/animals'
    output_path = './output/'
    
    print(f'Retrieving data from {data_path}')
    print(f'Outputing models to: {output_path}')

    # Define your transforms for RGB images.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Mean and std for RGB channels
    ])

    # Instantiate your custom dataset with the updated transform
    dataset = CustomImageDataset(directory=data_path, transform=transform)
    # val_dataset = CustomImageDataset(directory=val_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])

    # # Create the DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    print('Data Loaded')

    if num_pieces != 0:
        print('Obtaining index permutations...')
        selected_perms = get_perms(num_pieces, seed=seed)
        print('Permutations obtained')
    else:
        selected_perms = None

    # Initialize the generator and discriminator
    generator = FlexibleGenerator(output_size=image_size).to(device) 
    discriminator = FlexibleDiscriminator(input_size=image_size).to(device)

    # Run the training loop
    discriminator, generator, df_stats = train_loop(
        discriminator=discriminator, 
        generator=generator, 
        trainloader=train_loader, 
        valloader=val_loader, 
        lr=learning_rate, 
        num_epochs=num_epochs, 
        latent_dim=latent_dim,
        device=device,
        jigsaw=jigsaw,
        perms=selected_perms,
        num_pieces=num_pieces,
        print_batch=print_batch
        )

    model_date = f'{datetime.datetime.now():%Y%m%d_%H%m}_'
    model_name = '_j' + str(num_pieces)

    if save_data:
        # Save the trained generator model
        gen_path = output_path + '/models/' + 'g' + model_name + '.pth'
        torch.save(generator.state_dict(), gen_path)

        disc_path = output_path + '/models/' + 'd' + model_name + '.pth'
        torch.save(discriminator.state_dict(), disc_path)

        stats_path = output_path + '/stats/' + 'stats_pt' + model_name + '.csv'
        df_stats.to_csv(stats_path)
    
    return disc_path
        


def train_loop(discriminator, generator, trainloader, valloader, lr, num_epochs, latent_dim, device, jigsaw, perms, num_pieces, print_batch=None, print_im=False):
    """
    The main training loop for the GAN where both the generator and discriminator are trained.

    Args:
        discriminator (torch.nn.Module): The discriminator model.
        generator (torch.nn.Module): The generator model.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
        valloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        lr (float): Learning rate for optimizers.
        num_epochs (int): Total number of epochs to train for.
        latent_dim (int): Dimensionality of the latent space.
        device (torch.device): Device to run the training on.
        jigsaw (bool): Flag to indicate if jigsaw puzzle augmentation is used.
        perms (list): Permutations for jigsaw puzzle if used.
        num_pieces (int): Number of pieces to split each image for jigsaw puzzle.
        print_batch (int, optional): Interval of batches after which to print progress.
        print_im (bool, optional): Whether to print images during training.

    Returns:
        tuple: Trained discriminator and generator models, and a dataframe with training statistics.
    """
    #Initialise torch metrics
    auroc = AUROC(task='binary').to(device)
    f1_score_metric = F1Score(task='binary').to(device)
    accuracy_metric = Accuracy(task='binary').to(device)
    accuracy_real_metric = Accuracy(task='binary').to(device)
    accuracy_fake_metric = Accuracy(task='binary').to(device)

    
    if print_batch is None:
        print_batch = len(trainloader) + 1
    # Define loss and optimizers
    criterion = nn.BCELoss()
    jigsaw_criterion = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    
    epoch_auc_scores = []
    epoch_f1_scores = []
    epoch_accuracies = []
    
    val_epoch_auc_scores = []
    val_epoch_f1_scores = []
    val_epoch_accuracies = []

    epoch_accuracies_real = []
    epoch_accuracies_fake = []

    mean_real = []
    mean_fake = []

    d_losses = []
    g_losses = []

    alpha = 1
    beta = 0.2
    t0 = time.time()
    # Training loop
    for epoch in range(num_epochs):
        t1 = time.time()
        batch_auc_scores = []
        batch_f1_scores = []
        batch_accuracies = []
        
        batch_accuracies_real = []
        batch_accuracies_fake = []

        b_mean_real = []
        b_mean_fake = []
        
        batch_d_loss = []
        batch_g_loss = []

        for i, real_images in enumerate(trainloader):
            real_images = real_images.to(device) 
            batch_size = real_images.size(0)


            ### Train the Discriminator ###
            
            optimizer_d.zero_grad()
            # Train discriminator with real images
            label_rf_real = torch.ones(batch_size, 1).to(device) 
            pred_rf_real, _ = discriminator(real_images)
            pred_rf_real = pred_rf_real.view(-1, 1)
            loss_rf_real = criterion(pred_rf_real, label_rf_real)

            # Train discriminator with jigsaw shuffled real images
            if jigsaw:
                real_images_shuffled, label_jigsaw_real = shuffle_image(real_images, num_pieces, perms)
                real_images_shuffled = real_images_shuffled.to(device)
                label_jigsaw_real = label_jigsaw_real.to(device)
                _, pred_jigsaw_real = discriminator(real_images_shuffled)
                loss_jigsaw_real = jigsaw_criterion(pred_jigsaw_real, label_jigsaw_real)
            
            # Train discriminator with fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device) 
            fake_images = generator(noise).detach()
            label_rf_fake = torch.zeros(batch_size, 1).to(device) 
            pred_rf_fake, _ = discriminator(fake_images)
            pred_rf_fake = pred_rf_fake.view(-1, 1)
            loss_rf_fake_d = criterion(pred_rf_fake, label_rf_fake)

            # Add the losses
            loss_d = loss_rf_real + loss_rf_fake_d
            if jigsaw:
                loss_d += alpha*loss_jigsaw_real
            loss_d.backward()
            optimizer_d.step()


            ### Train the Generator ###

            optimizer_g.zero_grad()
            # Train generator with fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device) 
            fake_images = generator(noise)
            pred_rf_fake, _ = discriminator(fake_images)
            pred_rf_fake = pred_rf_fake.view(-1, 1)
            label_rf_real = torch.ones(batch_size, 1).to(device)   # Labels for generated images are 'real' to train the Generator
            loss_rf_fake_g = criterion(pred_rf_fake, label_rf_real)

            # Train the generator with jigsaw shuffled fake images
            if jigsaw:
                fake_images_shuffled, label_jigsaw_fake = shuffle_image(fake_images, num_pieces, perms)
                fake_images_shuffled = fake_images_shuffled.to(device)
                label_jigsaw_fake = label_jigsaw_fake.to(device)
                _, pred_jigsaw_fake = discriminator(fake_images_shuffled)
                loss_jigsaw_fake = jigsaw_criterion(pred_jigsaw_fake, label_jigsaw_fake)

            # Add the losses
            loss_g = loss_rf_fake_g
            if jigsaw:
                loss_g += beta*loss_jigsaw_fake
            loss_g.backward()
            optimizer_g.step()

            # Store discriminator outputs and labels for metrics
            with torch.no_grad():
                real_labels = torch.ones(batch_size, device=pred_rf_real.device)
                fake_labels = torch.zeros(batch_size, device=pred_rf_fake.device)
                real_outputs = pred_rf_real.squeeze()
                fake_outputs = pred_rf_fake.squeeze()

            # Compute metrics for each batch and store them
            b_mean_real.append(torch.mean(real_outputs).item())
            b_mean_fake.append(torch.mean(fake_outputs).item())
            combined_labels = torch.cat([real_labels, fake_labels])
            combined_outputs = torch.cat([real_outputs, fake_outputs])
            batch_auc_scores.append(auroc(combined_outputs, combined_labels).item())
            predictions = torch.cat([(real_outputs > 0.5).float(), (fake_outputs > 0.5).float()])
            batch_f1_scores.append(f1_score_metric(predictions, combined_labels).item())
            batch_accuracies.append(accuracy_metric(predictions, combined_labels).item())

            # Calculate real and fake metrics
            batch_accuracies_real.append(accuracy_real_metric((real_outputs > 0.5).float(), real_labels).item())
            batch_accuracies_fake.append(accuracy_fake_metric((fake_outputs > 0.5).float(), fake_labels).item())

            # # Store discriminator outputs and labels for metrics
            # with torch.no_grad():
            #     real_labels = torch.ones(batch_size).numpy()  # Real images have label 1
            #     fake_labels = torch.zeros(batch_size).numpy()  # Fake images have label 0
            #     real_outputs = pred_rf_real.squeeze().cpu().numpy()
            #     fake_outputs = pred_rf_fake.squeeze().cpu().numpy()

            # # Compute metrics for each batch and store them
            # b_mean_real.append(np.mean(real_outputs))
            # b_mean_fake.append(np.mean(fake_outputs))
            # batch_labels = np.concatenate([real_labels, fake_labels])
            # batch_outputs = np.concatenate([real_outputs, fake_outputs])
            # batch_auc_scores.append(roc_auc_score(batch_labels, batch_outputs))
            # batch_predictions = np.concatenate([(real_outputs > 0.5).astype(int), (fake_outputs > 0.5).astype(int)])
            # batch_f1_scores.append(f1_score(batch_labels, batch_predictions))
            # batch_accuracies.append(accuracy_score(batch_labels, batch_predictions))

            # # Calculate real and fake metrics
            # batch_accuracies_real.append(accuracy_score(real_labels, (real_outputs > 0.5).astype(int)))
            # batch_accuracies_fake.append(accuracy_score(fake_labels, (fake_outputs > 0.5).astype(int)))

            if jigsaw:
                batch_d_loss.append([loss_d.item(), loss_rf_real.item(), loss_rf_fake_d.item(), loss_jigsaw_real.item()])
                batch_g_loss.append([loss_g.item(), loss_rf_fake_g.item(), loss_jigsaw_fake.item()])
            else:
                batch_d_loss.append([loss_d.item(), loss_rf_real.item(), loss_rf_fake_d.item()])
                batch_g_loss.append(loss_g.item())

            if (i + 1) % print_batch == 0:
                if jigsaw:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(trainloader)}], '
                        f'D_real: {pred_rf_real.mean().item():.4f}, D_fake: {pred_rf_fake.mean().item():.4f}, '
                        f'Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}, '
                        f'Loss_jigsaw_real: {loss_jigsaw_real.item():.4f}, Loss_jigsaw_fake: {loss_jigsaw_fake.item():.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(trainloader)}], '
                        f'D_real: {pred_rf_real.mean().item():.4f}, D_fake: {pred_rf_fake.mean().item():.4f}, '
                        f'Loss_D: {loss_d.item():.4f}, Loss_G: {loss_g.item():.4f}')
        
        # Start of validation phase
        discriminator.eval()  # Set the discriminator to evaluation mode
        generator.eval()  # Set the generator to evaluation mode

        with torch.no_grad():  # No need to calculate gradients
            val_batch_auc_scores = []
            val_batch_f1_scores = []
            val_batch_accuracies = []

            for real_images in valloader:
                real_images = real_images.to(device)
                batch_size = real_images.size(0)

                # Evaluate discriminator with real images
                label_rf_real = torch.ones(batch_size, 1).to(device)
                pred_rf_real, _ = discriminator(real_images)
                pred_rf_real = pred_rf_real.view(-1, 1)

                # Generate fake images for evaluation
                noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
                fake_images = generator(noise)
                label_rf_fake = torch.zeros(batch_size, 1).to(device)
                pred_rf_fake, _ = discriminator(fake_images)
                pred_rf_fake = pred_rf_fake.view(-1, 1)

                # Calculate metrics for validation batch
                with torch.no_grad():
                    val_real_labels = torch.ones(batch_size, device=pred_rf_real.device)
                    val_fake_labels = torch.zeros(batch_size, device=pred_rf_fake.device)
                    val_real_outputs = pred_rf_real.squeeze()
                    val_fake_outputs = pred_rf_fake.squeeze()

                val_combined_labels = torch.cat([val_real_labels, val_fake_labels])
                val_combined_outputs = torch.cat([val_real_outputs, val_fake_outputs])
                val_batch_auc_scores.append(auroc(val_combined_outputs, val_combined_labels).item())
                val_predictions = torch.cat([(val_real_outputs > 0.5).float(), (val_fake_outputs > 0.5).float()])
                val_batch_f1_scores.append(f1_score_metric(val_predictions, val_combined_labels).item())
                val_batch_accuracies.append(accuracy_metric(val_predictions, val_combined_labels).item())


                # # Calculate metrics for validation batch
                # val_real_labels = torch.ones(batch_size).numpy()
                # val_fake_labels = torch.zeros(batch_size).numpy()
                # val_real_outputs = pred_rf_real.squeeze().cpu().numpy()
                # val_fake_outputs = pred_rf_fake.squeeze().cpu().numpy()
                # val_batch_labels = np.concatenate([val_real_labels, val_fake_labels])
                # val_batch_outputs = np.concatenate([val_real_outputs, val_fake_outputs])
                # val_batch_auc_scores.append(roc_auc_score(val_batch_labels, val_batch_outputs))
                # val_batch_predictions = np.concatenate([(val_real_outputs > 0.5).astype(int), (val_fake_outputs > 0.5).astype(int)])
                # val_batch_f1_scores.append(f1_score(val_batch_labels, val_batch_predictions))
                # val_batch_accuracies.append(accuracy_score(val_batch_labels, val_batch_predictions))

            # Calculate average validation metrics for the epoch
            val_epoch_auc = np.mean(val_batch_auc_scores)
            val_epoch_f1 = np.mean(val_batch_f1_scores)
            val_epoch_acc = np.mean(val_batch_accuracies)

            # Store validation metrics
            val_epoch_auc_scores.append(val_epoch_auc)
            val_epoch_f1_scores.append(val_epoch_f1)
            val_epoch_accuracies.append(val_epoch_acc)

        # Ensure the models are back to train mode
        discriminator.train()
        generator.train()

        # Compute average metric scores for the epoch
        epoch_auc = np.mean(batch_auc_scores)
        epoch_f1 = np.mean(batch_f1_scores)
        epoch_acc = np.mean(batch_accuracies)

        epoch_auc_scores.append(epoch_auc)
        epoch_f1_scores.append(epoch_f1)
        epoch_accuracies.append(epoch_acc)

        mean_real.append(np.mean(b_mean_real))
        mean_fake.append(np.mean(b_mean_fake))

        epoch_accuracies_real.append(np.mean(batch_accuracies_real))
        epoch_accuracies_fake .append(np.mean(batch_accuracies_fake))

        d_losses.append(np.mean(batch_d_loss, axis=0))
        g_losses.append(np.mean(batch_g_loss, axis=0))

        print(f'Epoch [{epoch+1}/{num_epochs}]: Discriminator Evaluation'
            f'\nTrain: \t\tAUC: {epoch_auc:.4f}, F1: {epoch_f1:.4f}, ACC: {epoch_acc:.4f}'
            f'\nValidation: \tAUC: {val_epoch_auc:.4f}, F1: {val_epoch_f1:.4f}, ACC: {val_epoch_acc:.4f}'
            f'\nEpoch Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t1))}, '
            f'Total Time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-t0))}')

        if print_im:
            print_images(generator, trainloader, latent_dim, device)

    d_losses = np.array(d_losses)
    g_losses = np.array(g_losses)
    
    data = {
        'epoch_accuracies': epoch_accuracies,
        'epoch_accuracies_real': epoch_accuracies_real,
        'epoch_accuracies_fake': epoch_accuracies_fake,
        'epoch_auc_scores': epoch_auc_scores,
        'epoch_f1_scores': epoch_f1_scores,
        'mean_real': mean_real,
        'mean_fake': mean_fake,
        'loss_d': d_losses[:, 0],
        'loss_rf_real': d_losses[:, 1],
        'loss_rf_fake_d': d_losses[:, 2],
    }
    if jigsaw:
        data['loss_jigsaw_real'] = d_losses[:, 3]
        data['loss_g'] = g_losses[:, 0]
        data['loss_rf_fake_g'] = g_losses[:, 1]
        data['loss_jigsaw_fake'] = g_losses[:, 2]
    else:
        data['loss_g'] = g_losses
    
    # Add validation metrics to the dataframe or data storage method before returning
    data['val_epoch_accuracies'] = val_epoch_accuracies
    data['val_epoch_auc_scores'] = val_epoch_auc_scores
    data['val_epoch_f1_scores'] = val_epoch_f1_scores


    return discriminator, generator, pd.DataFrame(data)

def print_images(generator, dataloader, latent_dim, device):
    """
    Generates and displays a grid of fake and real images for visual comparison.

    Args:
        generator (torch.nn.Module): The generator model that produces images.
        dataloader (torch.utils.data.DataLoader): DataLoader providing real images for comparison.
        latent_dim (int): Dimensionality of the latent space for the generator.
        device (torch.device): Device on which the generator operates.
    """
    # Generate and save sample images at the end of each epoch
    with torch.no_grad():
        fake_samples = generator(torch.randn(64, latent_dim, 1, 1).to(device))
        fake_samples = fake_samples.cpu()
        fake_grid = torchvision.utils.make_grid(fake_samples, padding=2, normalize=True)
        
        real_images_accumulated = []
        for real_images_batch in dataloader:
            real_images_accumulated.append(real_images_batch)
            if len(real_images_accumulated) * real_images_batch.size(0) >= 64:
                break
        real_images_accumulated = torch.cat(real_images_accumulated, dim=0)[:64]
        real_grid = torchvision.utils.make_grid(real_images_accumulated.cpu(), padding=2, normalize=True)
        
        # Plot the real and fake images for comparison
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(np.transpose(fake_grid, (1, 2, 0)))
        plt.title("Fake Images")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(np.transpose(real_grid, (1, 2, 0)))
        plt.title("Real Images")
        plt.axis('off')
        
        plt.show()
