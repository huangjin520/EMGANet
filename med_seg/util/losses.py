from util.parser import get_parser_with_args
from util.metrics import FocalLoss, dice_loss,TverskyLoss
from util.lovasz_softmax import lovasz_softmax
from util.MultiTverskyLoss import MultiTverskyLoss



def hybrid_loss(predictions, target,device):
    "Calculating the loss"
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=0)
    # tv=TverskyLoss(alpha=0.3, beta=0.7)
  

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target,device)
       # dice = lovasz_softmax(prediction, target)
        # dice=tv(prediction, target,device)
        
        loss += bce + dice

    return loss

