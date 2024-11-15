import torch.nn as nn
import torch

class FlexibleGenerator(nn.Module):
    """
    A generator model class for generating images from random noise using the transposed convolutional layers.
    This model can dynamically adjust its architecture based on the desired output image size.
    
    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_size (int): The desired width and height of the output image.
        image_channels (int): The number of channels in the output image.
    """
    def __init__(self, latent_dim=100, output_size=64, image_channels=3):
        super(FlexibleGenerator, self).__init__()
        
        modules = []         # List to hold the layers of the model
        current_size = 4     # Initial size of the image
        output_channels = output_size*8   # Initialize the number of output channels
        num_layers = 0      # Counter for the number of layers
        
        # Calculate the number of layers needed to reach the output size        
        while current_size < output_size:
            current_size *= 2
            num_layers += 1
        current_size = 4 

        # Build the layers of the generator
        for i in range(num_layers):
            if i == 0:  # First layer: Transposed convolution from the latent dimension
                modules.append(nn.ConvTranspose2d(latent_dim, output_channels, kernel_size=4, stride=1, padding=0))
            else:       # Subsequent layers: Halve the channels with each layer
                modules.append(nn.ConvTranspose2d(output_channels, output_channels // 2, kernel_size=4, stride=2, padding=1))
                output_channels //= 2
            
            modules.append(nn.BatchNorm2d(output_channels))
            modules.append(nn.ReLU(True))
        
        if current_size != output_size:   # Final layer to adjust to the desired image size and channel
            modules.append(nn.ConvTranspose2d(output_channels, image_channels, kernel_size=4, stride=2, padding=1, output_padding=output_size % 2))
        else:
            modules.append(nn.ConvTranspose2d(output_channels, image_channels, kernel_size=4, stride=2, padding=1))
        
        modules.append(nn.Tanh())
        
        self.model = nn.Sequential(*modules)

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): A batch of random noise vectors from the latent space.

        Returns:
            torch.Tensor: A batch of generated images.
        """
        return self.model(z)


class FlexibleDiscriminator(nn.Module):
    """
    A discriminator model class for distinguishing real images from generated ones, 
    and optionally performing jigsaw puzzle recognition as a form of auxiliary task to improve training stability.
    
    Args:
        input_size (int): Width and height of the input images.
        image_channels (int): Number of channels in the input images.
        num_classes (int): Number of classes for the jigsaw puzzle task (used if jigsaw puzzle recognition is enabled).
    """
    def __init__(self, input_size=64, image_channels=3, num_classes=30):
        super(FlexibleDiscriminator, self).__init__()
        
        modules = []
        current_size = input_size
        input_channels = 32
        i=0

        # Create convolutional layers to progressively downsample the image
        while current_size > 7:
            if current_size==input_size:  # First layer uses the number of image channels
                modules.append(nn.Conv2d(image_channels, input_channels * 2, 4, 2, 1))
            else:
                modules.append(nn.Conv2d(input_channels, input_channels * 2, 4, 2, 1))  # Subsequent layers double the number of channels from the previous layer
                modules.append(nn.BatchNorm2d(input_channels * 2))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
            input_channels *= 2
            current_size //= 2
        
        self.model = nn.Sequential(*modules)  # Main feature extractor model

        # Output layers for real/fake classification and jigsaw classification
        self.real_fake = nn.Sequential(
            nn.Conv2d(input_channels, 1, 4, 1, 0),
            nn.Sigmoid()
        )

        self.jigsaw = nn.Sequential(
            nn.Conv2d(input_channels, num_classes, 4, 1, 0),
            nn.Sigmoid()
        )     

    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x (torch.Tensor): Input image batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the real/fake predictions and jigsaw puzzle class predictions.
        """    
        out = self.model(x)
        rf = self.real_fake(out)
        jig = self.jigsaw(out)
        
        return rf, jig.view(x.shape[0], -1)

class FlexibleGANForSegmentation(nn.Module):
    """
    A GAN model specifically tailored for segmentation tasks using a pre-trained discriminator to further refine segmentation capabilities.
    
    Args:
        model_filepath (str, optional): Path to a pre-trained discriminator model.
        output_size (int): The target output size of the segmentation map.
    """
    def __init__(self, model_filepath = None, output_size=64):
        super(FlexibleGANForSegmentation, self).__init__()
        
        modules = []
        input_channels = output_size * 8

        # Initialize the discriminator (possibly with pre-trained weights)
        discriminator = FlexibleDiscriminator(input_size=output_size)
        if model_filepath:
            discriminator.load_state_dict(torch.load(model_filepath, map_location=torch.device('cpu')))
            # for param in discriminator.parameters():
            #     param.requires_grad = False

        modules.append(discriminator.model)
        while input_channels > 64:  # Build up-sampling layers to increase the resolution of the output
            modules.append(nn.ConvTranspose2d(input_channels, input_channels // 2, 4, 2, 1))
            modules.append(nn.BatchNorm2d(input_channels // 2))
            modules.append(nn.LeakyReLU(inplace=True))
            input_channels //= 2
        modules.append(nn.ConvTranspose2d(input_channels, 3, 4, 2, 1))
        modules.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        """
        Forward pass of the GAN for segmentation.
        
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Segmented output tensor.
        """
        out = self.model(x)
        return out
