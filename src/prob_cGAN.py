import yaml
import omegaconf
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset

def load_config(config_path: str):
    """
    Load configuration from a YAML file and convert to OmegaConf format.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        config = omegaconf.OmegaConf.create(config)
    return config

class Custom_Dataset(Dataset):
    """
    Custom Dataset
    y= target
    x= input
    """
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.n_samples

def norm_precip(target_data,a):
    # normalizing target data
    target_data = np.log10(target_data + a)-np.log10(a)
    return target_data
def reverse_norm_precip(norm_data, a):
    # Reversing the normalization
    original_data = (10 ** (norm_data + np.log10(a))) - a
    return original_data

def enable_dropout_only(model):
    """Enable dropout layers but keep batchnorm (and others) in eval mode."""
    # Put model in eval first (disable BN updates)
    model.eval()
    # Then turn on dropout-type modules
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()   # enable dropout sampling
    return model

def calc_prediction(generator,model_path,testing_loader,add_random_channel=False,seed=1,config=None):
    generator.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    generator = enable_dropout_only(generator)

    predictions = []
    targets = []
    inputs = []
    for batch in testing_loader:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if add_random_channel:
            if config.training.get('perlin', None) is not None:
                random_channel = tf.generate_perlin_noise(batch[0][:, :1, ...].shape, config.training.perlin).to(batch[0].device)
                random_channel = random_channel.repeat(batch[0].size(0), 1, 1, 1) * config.training.noise_weight
            else:
                random_channel = (torch.rand_like(batch[0][:, :1, ...])*config.training.noise_weight)
            batch[0] = torch.cat((batch[0], random_channel), dim=1)
        pred_test = generator(batch[0]).detach().cpu().numpy()
        predictions.append(pred_test)
        targets.append(batch[1].detach().cpu().numpy())
        inputs.append(batch[0].detach().cpu().numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0), np.concatenate(inputs, axis=0)


class ResNetBlockWithSE3D(nn.Module):
    """
    ResNet-style block with integrated SE attention
    """
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=16,padding_mode='zeros'):
        super(ResNetBlockWithSE3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # SE block after second convolution
        self.se_block = EfficientSEBlock3D(out_channels, reduction)
        
        # Skip connection
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False, padding_mode=padding_mode),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE attention
        out = self.se_block(out)
        
        # Skip connection
        out += self.skip_connection(x)
        out = F.relu(out)
        
        return out
    
class EfficientSEBlock3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, C, T, H, W)
        squeezed = x.mean(dim=[3, 4])  # (B, C, T)
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeezed))))  # (B, C, T)
        return x * excitation.unsqueeze(-1).unsqueeze(-1)  # broadcast to (B, C, T, H, W)
    

class ResNetBlock3D(nn.Module): 
    """
    Block with 2 convolutional layers, with normalisation and ReLu.

    Input: in/out channels, one batch x: torchtensor
    Outut: results y + input data x (projected)
    """
    def __init__(self,in_channels: int,out_channels: int,padding_mode='zeros'):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.projection = None
        if in_channels != out_channels:
            self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding_mode=padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.projection is None:
            shortcut = x
        else:
            shortcut = self.projection(x)
        return y + shortcut
    
class ResizeConv3D_TStatic(nn.Module):
    def __init__(self, in_channels, out_channels, upscale=(1, 2, 2),
                 padding_mode='zeros'):
        super().__init__()
        self.upscale = tuple(upscale)
        self.padding_mode = padding_mode
        self.upsample = nn.Upsample(scale_factor=self.upscale, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class Unet_ensemble_3D_alldropout(nn.Module):
    def __init__(
        self,
        T_in,
        in_channels=1,
        in_features_G=[32, 64, 128, 256],
        padding_mode='zeros',
        SE_encoder=False,
        SE_decoder=False,
        dropout_cfg=None,    
    ):
        super(Unet_ensemble_3D_alldropout, self).__init__()

        # --------- DROPOUT CONFIG FIX ---------
        # default values if dropout_cfg is missing or partial
        dropout_cfg = dropout_cfg or {}
        self.e1_type = dropout_cfg.get("e1", "3D")
        self.other_type = dropout_cfg.get("other", "overall")
        self.p_e1 = dropout_cfg.get("p_e1", 0.01)
        self.p_other = dropout_cfg.get("p_other", 0.001)

        # First encoder dropout
        if self.e1_type == '3D':
            self.dropout_e1 = nn.Dropout3d(p=self.p_e1)
        else:
            self.dropout_e1 = nn.Dropout(p=self.p_e1)

        # Remaining encoder & decoder dropout
        if self.other_type == '3D':
            self.dropout_other = nn.Dropout3d(p=self.p_other)
        else:
            self.dropout_other = nn.Dropout(p=self.p_other)
        # ---------------------------------------

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.poolingtstatic = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Encoder
        for feature in in_features_G:
            block = (ResNetBlockWithSE3D if SE_encoder else ResNetBlock3D)(in_channels, feature, padding_mode=padding_mode)
            self.encoder.append(block)
            in_channels = feature

        # Decoder
        for feature in reversed(in_features_G):
            layer = in_features_G.index(feature) + 1
            if T_in / layer < 2:
                self.decoder.append(ResizeConv3D_TStatic(feature * 2, feature, upscale=(1, 2, 2), padding_mode=padding_mode))
            else:
                self.decoder.append(ResizeConv3D_TStatic(feature * 2, feature, upscale=(2, 2, 2), padding_mode=padding_mode))

            block = (ResNetBlockWithSE3D if SE_decoder else ResNetBlock3D)(feature * 2, feature, padding_mode=padding_mode)
            self.decoder.append(block)

        self.bottleneck = ResNetBlock3D(in_features_G[-1], in_features_G[-1] * 2,padding_mode=padding_mode)
        self.time_reduction = nn.Conv3d(in_features_G[0], in_features_G[0], kernel_size=(T_in, 1, 1))
        self.final = nn.Sequential(
            nn.Conv3d(in_features_G[0], 1, kernel_size=1),
            nn.LeakyReLU(0.01)
        )
    def forward(self,x):
        skipconnect =[]
        for i,encode in enumerate(self.encoder):
            x = encode(x)
            time_dim = x.shape[2]
            if i ==0:
                x = self.dropout_e1(x)
            else:
                x = self.dropout_other(x)
            skipconnect.append(x)
            if time_dim==1:
                x = self.poolingtstatic(x)
            else:
                x = self.pooling(x)

        x = self.bottleneck(x)

        skipconnect=skipconnect[::-1]
        for idx in range(0,len(self.decoder),2):
            x =self.decoder[idx](x)
            x = self.dropout_other(x)
            concat_x_skip = torch.cat((skipconnect[idx//2],x),dim=1)
            x = self.decoder[idx+1](concat_x_skip)
       
        x = self.time_reduction(x)  # Output: (B, out_channels, 1, H, W)   
        x = self.final(x)
        return(x)