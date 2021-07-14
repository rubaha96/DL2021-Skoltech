import torch

def calc_val_data(preds, masks, num_classes):
    preds = torch.argmax(preds, dim=1)

    intersection = torch.zeros(preds.shape[0], num_classes) # TODO: calc intersection for each class
    union = torch.zeros(preds.shape[0], num_classes) # TODO: calc union for each class
    target = torch.zeros(preds.shape[0], num_classes) # TODO: calc number of pixels in groundtruth mask per class
    # Output shapes: B x num_classes
    for i in range(num_classes):
        intersection[:, i] = torch.sum((preds == i) & (masks == i), (1, 2))
        union[:, i] = torch.sum((preds == i) | (masks == i), (1, 2))
        target[:, i] = torch.sum((masks == i), (1, 2))

    assert isinstance(intersection, torch.Tensor), 'Output should be a tensor'
    assert isinstance(union, torch.Tensor), 'Output should be a tensor'
    assert isinstance(target, torch.Tensor), 'Output should be a tensor'

    assert intersection.shape == union.shape == target.shape, 'Wrong output shape'
    assert union.shape[0] == masks.shape[0] and union.shape[1] == num_classes, 'Wrong output shape'

    return intersection, union, target

def calc_val_loss(intersection, union, target, eps = 1e-7):
    mean_iou = torch.mean((intersection + eps)/(union + eps)) # TODO: calc mean class iou
    mean_class_rec = torch.mean((intersection + eps)/(target + eps)) # TODO: calc mean class recall
    mean_acc = torch.mean(torch.sum(intersection + eps, 1)/torch.sum(target + eps, 1)) # TODO: calc mean accuracy

    return mean_iou, mean_class_rec, mean_acc