import torch
import torch.nn as nn

def normalize(data):
    normed_data = nn.Sigmoid()(data)
    return normed_data


def normalize_data(data):

    if data.shape[0] == 1:
        return data
    
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-8)

    return (data - mean) / std


def KL_divergence(mu, sigma, sigma2 = 1, min_value=1e-8):
    """
    Calculate the KL divergence between two Gaussian distributions.
    """

    batch_size = mu.size()[0]
    J = mu.size()[1]

    mu_diff = (mu) ** 2
    var1 = sigma ** 2
    var2 = sigma2 ** 2
    var_frac = var1 / var2
    diff_var_frac = mu_diff / var2

    var_frac = torch.clamp(var_frac, min=min_value)

    term1 = torch.sum(torch.log(var_frac)) / batch_size
    term2 = torch.sum(var_frac) / batch_size
    term3 = torch.sum(diff_var_frac) / batch_size

    return - 0.5 * (term1 - term2 - term3 + J)

def reconstruct_loss(output, target, zero_weight=0.1, l1_lambda=0.01):
    """
    Calculate the reconstruction loss with L1 regularization and zero-weighting.
    Parameters:
        output (torch.Tensor): The output from the decoder, shape (N, channels, H, W).
        target (torch.Tensor): The target feature map, shape (N, channels, H, W).
        zero_weight (float): Weight for the zero elements in the target.
        l1_lambda (float): Regularization strength for L1 loss.
        
    Returns:
        torch.Tensor: The total reconstruction loss.
        
    """
    if zero_weight is None:
        mse_loss = nn.functional.mse_loss(output, target)
        return mse_loss
    
    non_zero_mask = (target != 0).float()
    zero_mask = (target == 0).float()
    
    non_zero_mse_loss = nn.functional.mse_loss(output * non_zero_mask, target * non_zero_mask, reduction='sum') / (non_zero_mask.sum() + 1e-8)
    zero_mse_loss = nn.functional.mse_loss(output * zero_mask, target * zero_mask, reduction='sum') / (zero_mask.sum() + 1e-8)
    
    mse_loss = (1 - zero_weight) * non_zero_mse_loss + zero_weight * zero_mse_loss

    # L1 regularization to encourage sparsity
    l1_loss = torch.mean(torch.abs(output))

    # Total loss
    total_loss = mse_loss + l1_lambda * l1_loss
    
    return total_loss



def get_ego_features(feature, record_len):

    record_len_list = record_len.tolist()
    indices = [0]
    for i in range(len(record_len_list) - 1):
        indices.append(indices[-1] + record_len_list[i])

    ego_features = feature[indices]

    with torch.no_grad():
        spatial_mask = torch.any(feature, dim=1).float().unsqueeze(1).to(feature.device)

    return ego_features, indices, spatial_mask



def quantize_tensor(x: torch.Tensor, bits: int = 8, zero_as_zero: bool = False, return_quantized: bool = False) -> torch.Tensor:
    """
    Quantize the input tensor to fixed-point while preserving positions that are exactly zero.

    Parameters:
        x (torch.Tensor): Input tensor with values in [0, 1].
        bits (int): Number of quantization bits (e.g., 8).
        zero_as_zero (bool): If True, keep original zero-valued positions exactly zero.
        return_quantized (bool): If True, return a tuple (quantized_tensor, dequantized_tensor);
                                 otherwise return only the dequantized tensor.

    Returns:
        If return_quantized is True:
            (torch.Tensor, torch.Tensor): quantized integer tensor and dequantized float tensor.
        Else:
            torch.Tensor: dequantized float tensor.
    """
    scale = 2 ** bits - 1

    if zero_as_zero:
        zeros_mask = (x == 0)
        non_zero_part = torch.where(zeros_mask, torch.zeros_like(x) + 1e-6, x)
        x_scaled = torch.round(non_zero_part * scale).clamp(0, scale)
    else:
        x_scaled = torch.round(x * scale).clamp(0, scale)

    # Dequantize
    x_dequantized = x_scaled / scale

    if zero_as_zero:
        x_dequantized = torch.where(zeros_mask, torch.zeros_like(x_dequantized), x_dequantized)

    if return_quantized:
        return x_scaled.to(torch.uint8), x_dequantized
    else:
        return x_dequantized


def topk_mask(mask: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    """
    Preserve the top 'ratio' proportion of values in each sample of the mask and set the rest to zero.
    Guarantees exactly k = ceil(total * ratio) elements are kept per sample.

    Args:
        mask (torch.Tensor): Tensor of shape (N, C, H, W), typically (N, 1, H, W).
        ratio (float): Fraction of elements to retain per sample (e.g., 0.1 means keep 10%).

    Returns:
        torch.Tensor: Tensor with the same shape as mask where only the top ratio values are retained.
    """
    N, C, H, W = mask.shape
    device = mask.device
    flat_mask = mask.view(N, -1)  # (N, H*W)
    total = flat_mask.shape[1]
    k = max(1, int(torch.ceil(torch.tensor(total * ratio)).item()))  # 计算 k

    result = torch.zeros_like(flat_mask)

    for i in range(N):
        values_i = flat_mask[i]  # (total,)
        
        # Step 1: 取 top-k 值
        topk_vals, _ = torch.topk(values_i, k=k, largest=True, sorted=True)
        threshold_val = topk_vals[-1]

        # Step 2: 所有大于 threshold_val 的都保留
        selected_mask = (values_i > threshold_val).bool()

        # Step 3: 统计当前个数
        current_count = selected_mask.sum().item()

        # Step 4: 如果还少于 k，从等于 threshold_val 的值中随机选补上
        if current_count < k:
            equal_mask = (values_i == threshold_val).bool()
            equal_indices = torch.nonzero(equal_mask, as_tuple=False).squeeze()
            need_more = k - current_count
            if equal_indices.dim() == 0:
                equal_indices = equal_indices.unsqueeze(0)
            rand_indices = equal_indices[torch.randperm(len(equal_indices), device=device)[:need_more]]
            selected_mask[rand_indices] = True

        elif current_count > k:
            # Step 5: 如果多于 k，从等于 threshold_val 的值中随机删掉多余的
            equal_mask = (values_i == threshold_val).bool()
            equal_indices = torch.nonzero(equal_mask, as_tuple=False).squeeze()
            remove_num = current_count - k
            if equal_indices.dim() == 0:
                equal_indices = equal_indices.unsqueeze(0)
            rand_indices = equal_indices[torch.randperm(len(equal_indices), device=device)[:remove_num]]
            selected_mask[rand_indices] = False

        result[i] = selected_mask.float()

    output = flat_mask * result
    return output.view_as(mask)


def threshold_sampling(threshold: float = 0.5, min_threshold: float = 0.1, batch_size: int = 1, p=0.6, device=None) -> torch.Tensor:
    """
    Samples values according to a specified probability distribution:
    - With p probability, returns the threshold value threshold
    - With 1-p probability, returns a uniform random value between min_threshold and 1
    
    Args:
        threshold (float): Threshold value, default is 0.1
        batch_size (int): Number of samples to generate, default is 1
    
    Returns:
        torch.Tensor: Tensor containing the sampled values
    """
    # Validate that the threshold min_threshold and threshold are positive
    if not (1 >= min_threshold >= 0 and 1 >= threshold >= 0):
        raise ValueError("Threshold min_threshold and threshold must be a positive number")
    
    if p <= 0:
        raise ValueError("Threshold p must be a positive number")
    
    # Generate binary choices (0 or 1) where 0 indicates choosing min_threshold and 1 indicates choosing a random value
    # Probability of choosing 0 is 0.8, probability of choosing 1 is 1-p
    choices = torch.bernoulli(torch.full((batch_size,), 1-p))
    
    # # Generate uniform random values between 0.0 and min_threshold
    # random_values = torch.rand(batch_size) * min_threshold

    # Generate uniform random values between min_threshold and 1
    random_values = min_threshold + (1.0 - min_threshold) * torch.rand(batch_size)
    
    # Determine the final sampled values based on the choices
    # If choice is 0, return min_threshold; if choice is 1, return the random value
    samples = torch.where(choices == 0, torch.tensor(threshold), random_values)
    
    if device is not None:
        # Move the sampled values to the specified device
        samples = samples.to(device)
    return samples