from .BetaVAE import *
from .MKMMD_VAE import *
from .InfoVAE import *
from .BetaVAE_new import *
from .MMD_WAE import *
from .MSSIM_VAE import *
from .S_WAE import *
from .MKMMD_InfoVAE import *
from .AAE import *

vae_models={'BetaVAE':Lit_BetaVAE,
'MKMMD_VAE': Lit_MKMMD_VAE,
'InfoVAE': Lit_InfoVAE,
'BetaVAE_new':Lit_BetaVAE_new,
'MMD_WAE': Lit_MMD_WAE,
'MSSIM_VAE': Lit_MSSIM_VAE,
'S_WAE': Lit_S_WAE,
'MKMMD_InfoVAE': Lit_MKMMD_InfoVAE,
'AAE': Lit_AAE}