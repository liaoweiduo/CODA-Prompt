from __future__ import absolute_import

from .dataloader import iCIFAR100, iCIFAR10, iIMAGENET_R, iDOMAIN_NET, iCGQA, iCOBJ

from .cfst_dataset import CGQA, COBJ

__all__ = ('iCIFAR100','iCIFAR10','iIMAGENET_R','iDOMAIN_NET', 'CGQA', 'COBJ', 'iCGQA', 'iCOBJ')