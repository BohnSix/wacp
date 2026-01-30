import torch
import torch.nn as nn

from opencood.extensions.coib_encoder_baseline import CoIBEncoder, CoIBDecoder
from opencood.extensions.networks import ScaleLayer, AdaptiveScaleLayer, Masknet, init_linear_weight
from opencood.extensions.utils import normalize, normalize_data, reconstruct_loss, get_ego_features, topk_mask, quantize_tensor, threshold_sampling

class IBCommunication(nn.Module):
    
    def __init__(self, channels=64, encoding_dim=256, decoding_dim=256, output_size=(200, 704), use_transforms=True, use_ego_feature=True, verbose=False, out_channels=256):
        """
        Initialize the ResNet-based autoencoder for sparse inputs.
        
        Parameters:
            channels (int): Number of input/output channels, default is 64.
            encoding_dim (int): Dimension of the latent feature vector, fixed to 256.
            output_size (tuple): Spatial size of the output feature map, default is (200, 704).
        """
        super(IBCommunication, self).__init__()

        # Store output size for reshaping
        self.channels = channels
        self.use_transforms = use_transforms
        self.verbose = verbose
        self.use_ego_feature = use_ego_feature

        self.initial_channels = 256  # Initial channels after reshaping
        self.initial_size = (output_size[0] // 8, output_size[1] // 8)  # Dynamic initial spatial size

        # Encoder: Modified CoIBEncoder
        self.encoder = CoIBEncoder(input_dim = channels, output_dim = encoding_dim)

        # Decoder: Reconstruct the feature map using transposed convolutions
        self.decoder = CoIBDecoder(input_dim = encoding_dim, output_dim = channels, initial_channels = self.initial_channels, initial_size = self.initial_size)

        self.mu_net = nn.Sequential(
            nn.Linear(encoding_dim, decoding_dim),
            AdaptiveScaleLayer(dim=decoding_dim),
        )

        self.std_net = nn.Sequential(
            nn.Linear(encoding_dim, decoding_dim),
            AdaptiveScaleLayer(dim=decoding_dim),
            nn.Softplus(),
        )

        init_linear_weight(self.mu_net[0], bias_value=0.0)
        init_linear_weight(self.std_net[0], bias_value=0.0)

        # self.adapt_layer_in = nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1)
        self.adapt_layer_in = Masknet(channels, out_channels)
        self.adapt_layer_out = nn.ConvTranspose2d(2, 1, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x, record_len, cls_head, data_dict, required_KL=True):

        ### spatial mask
        quantize_spatial_mask = self.get_mask(x, cls_head, data_dict, bits=4)

        ### coib communication
        _, reconstructed_features, KL_value, RL_value = self.autoencoder_forward(x, mask=quantize_spatial_mask, required_KL=required_KL)

        if self.use_ego_feature:
            ### For fix some bugs
            reconstructed_features =  reconstructed_features * 1
            # reconstructed_features =  reconstructed_features * quantize_spatial_mask
            
            ### Get ego features and indices
            ego_features, indices, _ = get_ego_features(x, record_len)
            ### Replace the features at the ego indices
            reconstructed_features[indices] = ego_features

        return reconstructed_features, KL_value, RL_value
    
    def get_mask(self, x, cls_head, data_dict, bits=4):
        ### init spatial mask
        spatial_mask = self.adapt_layer_out(cls_head(self.adapt_layer_in(x))).sigmoid()
        
        ### threshold for spatial mask
        epoch = data_dict.get('epoch', None)
        max_epoch = data_dict.get('max_epoch', None)

        if epoch is None or max_epoch is None:
            ratio = 0.1
            if self.verbose:
                print(f"epoch or max_epoch is None, using default ratio {ratio}")
        else:
            min_threshold = 0.1
            ratio = torch.clamp(torch.linspace(1, -1, max_epoch), min=min_threshold)[epoch].item()
            if self.training:
                ratio = threshold_sampling(ratio, device=spatial_mask.device)
            
            if self.verbose:
                print(f"epoch: {epoch}, max_epoch: {max_epoch}, ratio: {ratio}")

        spatial_mask_threshold = topk_mask(spatial_mask, ratio=ratio)

        ### quantization for spatial mask by gradient estimation
        quantize_spatial_mask = (quantize_tensor(spatial_mask_threshold, bits=bits, zero_as_zero=False) - spatial_mask).detach() + spatial_mask # torch.Size([3, 1, 200, 704])

        return quantize_spatial_mask


    def autoencoder_forward(self, x, mask=None, required_KL=True, required_RL=False):
        """
        Forward pass for the entire autoencoder.
        
        Parameters:
            x (torch.Tensor): Input feature map, shape (N, channels, H, W).
        
        Returns:
            torch.Tensor: Reconstructed feature map, shape (N, channels, H, W).
        """

        if self.use_transforms:
            adapt_x = normalize(x)
        else:
            adapt_x = x

        encoded_feature = self.encoder(adapt_x)

        # mu = self.mu_net(normalize_data(encoded_feature))
        mu = self.mu_net(encoded_feature)
        std = self.std_net(encoded_feature)
        std = torch.clamp(std, min=1e-6)  # Ensure std is not zero
        
        if self.verbose:
            print(f'mu: {mu.detach().mean(1)}, mu std: {mu.detach().std(1)}')
            print(f'std: {std.detach().mean(1)}, std std: {std.detach().std(1)}')

        if self.training:
            eps = torch.randn_like(std)

            received_feature = mu + torch.mul(eps, std)
        else:
            received_feature = mu

        decoded_feature = self.decoder(received_feature, mask)

        if required_KL:
            KL = torch.mean(-0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=1), dim=0)

            if required_RL:
                RL = reconstruct_loss(decoded_feature, x)
            else:
                RL = None
            return encoded_feature, decoded_feature, KL, RL
        else:
            return encoded_feature, decoded_feature

if __name__ == "__main__":
    model = IBCommunication(channels=64, encoding_dim=256, output_size=(200, 704))

    batch_size = 2
    input_tensor = torch.randn(batch_size, 64, 200, 704)
    input_tensor[input_tensor.abs() < 2.0] = 0.0 

    reconencoded_feature, decoded_feature, KL, RL = model(input_tensor, mask=torch.randn(batch_size, 1, 200, 704))

    loss = model.reconstruct_loss(input_tensor, decoded_feature)
    print("Loss:", loss.item())

    # Print output shapes
    print("Input feature shape:", input_tensor.shape)  # expected (batch_size, 64, 200, 704)
    print("Reconstructed feature shape:", decoded_feature.shape)  # expected (batch_size, 64, 200, 704)