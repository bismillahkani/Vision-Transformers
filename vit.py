import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Split the images into patches and embed them 
    
    Parameters
    ----------
    img_size : int
        Size of the image (it is a square)
    
    patch_size : int
        Size of the patch (it is a square)
    
    in_chans : int
        Number of channels
    
    embed_dim : int
        Embedding dimension 
    
    Attributes
    ----------
    n_patches : int
        Number of patches inside the image
    
    proj : nn.Conv2d
        Convolutional layer that does splitting into patches and embedding
    """
    
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__() 
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        # 2D-Conv for splitting into patches and embedding 
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """ Run forward pass 
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (n_batches, in_chans, img_size, img_size) 
        
        Returns
        -------
        torch.Tensor
            Shape (n_batches, n_patches, embed_dim) """
        
        x = self.proj(x) # (n_batches, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2) # (n_batches, embed_dim, n_patches)
        x = x.transpose(1,2) # (n_batches, n_patches, embed_dim)
    
        return x

class Attention(nn.Module): 
    """ Attention mechanism 
    
    Parameters
    ----------
    dim : int
        Input and output dimension of per token features
    
    n_heads : int
        Number of attention heads
    
    qkv_bias : bool
        If True include bias term in query, key and value projection
    
    attn_p : float
        Dropout rate applied to q, k, v tensors
    
    proj_p : float
        Dropout rate applied to output tensors
        
    Attributes
    ----------
    scale : float
        Normalizing constant for the dot product
    
    qkv : nn.Linear
        Linear projection of query, key and value
    
    proj : nn.Linear
        Linear projection of concatenated head
    
    attn_drop, proj_drop : nn.Dropout
        Dropout layers
    """
    
    def __init__(self, dim=768, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5 # This is the scale factor to prevent feeding large values into softmax

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias) # Either use separate linear projection for q,k,v or all together
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """ Run forward pass 
        
        Paramaters
        ----------
        x : torch.Tensor
            Shape (n_batches, n_patches+1, dim) - The +1 is for the class token 
            
        Returns
        -------
        torch.Tensor
            Shape (n_batches, n_patches+1, dim)
        """
        
        n_batches, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError
        
        qkv = self.qkv(x) # (n_batches, n_patches+1, dim * 3)
        qkv = qkv.reshape(n_batches, n_tokens, 3, self.n_heads, self.head_dim) 
                            # (n_batches, n_patches+1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_batches, n_heads, npatches+1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2,-1) # (3, n_batches, n_heads, head_dim, n_patches+1)
        dp = (q @ k_t) * self.scale # (3, n_batches, n_heads, n_patches+1, n_patches+1) DOT-PRODUCT Attention
        attn = dp.softmax(dim=-1) # Weights 
        attn = self.attn_drop(attn)  

        weighted_avg = attn @ v # (n_batches, n_heads, n_patches+1, head_dim) # Weighted average
        weighted_avg = weighted_avg.transpose(1,2) # (n_batches, n_patches+1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_batches, n_patches+1, dim)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x

class MLP(nn.Module):
    """ Multilayer perceptron 
    
    Parameters
    ----------    
    in_features : int
        Number of input features 
        
    hidden_features : int 
        Number of nodes in hidden layers 
    
    out_features : int 
        Number of output features 
    
    Attibutes
    ---------
    fc : nn.Linear
        First linear layer 
    
    act : nn.GELU
        GELU activation layers - Gaussian Error Linear Activation Unit
        
    fc2 : nn.Linear
        Second linear layer
    
    drop : nn.Dropout 
        Dropout layers 
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """ Run forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (n_batches, n_patches+1, in_features)
        
        Returns
        -------
        torch.Tensor
            Shape (n_batches, n_patches+1, out_features)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    """ Transformer Block 
    
    Parameters
    ----------
    dim : int
        Embedding dimension
    
    n_heads : int
        Number of heads
    
    mlp_ratio : float
        Determines the hidden size of the MLP module with respect to dim
    
    qkv_bias : bool
        If True include bias in q, k, v
    
    p, attn_p : float
        Dropout rate
    
    Attributes
    ----------
    norm1, norm2 : nn.LayerNorm
        Layer normalization
    
    attn : Attention
        Attention module
    
    mlp : MLP
        MLP Module
    """
    def __init__(self,dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # eps values was choses to match pre-trained model
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_features, dim)

    def forward(self, x):
        """ Run forward pass
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (n_batches, n_patches+1, dim)
        
        Returns
        -------
        torch.Tensor
            Shape (n_batches, n_patches+1, dim)
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    """ Simplified implementation of Vision Transformer Transformer
    
    Parameters
    ----------
    img_size : int
        Input image size (it is a square)
    
    patch_size : int
        Patch size (it is a square)
    
    in_chans : int
        Number of channels in input image
    
    n_classes : int
        Number of output classes
    
    embed_dim : int
        Embedding dimension
    
    depth  : int
        Number of transformer layers
    
    n_heads : int
        Number of attention heads
    
    mlp_ratio : int
        Determines the hidden dimension of MLP Module
    
    qkv_bias : bool
        If True, bias terms is included in q,k,v
    
    p, attn_p : float
        Dropout probability
    
    Attributes
    ----------    
    patch_embed : PatchEmbed
        Patch Embedding
    
    cls_token : nn.Parameter
        Learnable parameters of the first token of the sequence.
        This cls_token is used for classification
    
    pos_embed : nn.Parameter
        Positional embedding of cls_token and all patches
    
    pos_drop : nn.Dropout
        Droput layer

    blocks : nn.Modulelist
        List of block modules

    norm : nn.LayerNorm
        Layer normalization    
    """
    def __init__(self,
                img_size=224,
                patch_size=16,
                in_chans=3,
                n_classes=1000,
                embed_dim=768,
                depth=12,
                n_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                p=0.,
                attn_p=0.):
        super().__init__()

        self.patch_embed = PatchEmbed(
                                    img_size=img_size,
                                    patch_size=patch_size,
                                    in_chans=in_chans,
                                    embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim)) # Initialize class token with zeros. 
        self.pos_embed = nn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches, embed_dim)) # Learnable positional embedding
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim,
                        n_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        p=p,
                        attn_p=attn_p)
                
                for _ in range(depth) # depth is the number of encoder blocks
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """ Run the forward pass
        Parameters
        ----------
        x : torch.Tensor
            Shape (n_batches, in_chans, img_size, img_size)
            
        Returns
        -------
        logits : torch.Tensor
            Logits for n_classes (n_batches, n_classes)
        """
        n_batches = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_batches, -1, -1) # (n_batches, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1) # (n_batches, 1+n_patches, embed_dim)
        x = x + self.pos_embed # (n_batches, 1+n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        cls_token_final = x[:,0] 
        x = self.head(cls_token_final)

        return x

    
            
    



    
