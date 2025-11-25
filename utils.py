import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self,
                 ignore_index=-1,
                 prompt_num=2):
        super(ContrastiveLoss, self).__init__()

        self.ignore_label = ignore_index
        self.prompt_num = prompt_num

    def sample_anchor(self, features, label, ignore, n_view):
        batch_size, feat_dim = features.shape[0], features.shape[-1]

        classes = []
        total_classes = 0
        for i in range(batch_size):
            i_label = label[i]

            i_classes = torch.unique(i_label)
            i_classes = [x for x in i_classes if x not in ignore]

            classes.append(i_classes)
            total_classes += len(i_classes)
        if total_classes == 0:
            return None, None

        anc_features = []
        anc_labels = []

        for i in range(batch_size):
            i_label = label[i]
            i_classes = classes[i]

            for cls_id in i_classes:
                indices = (cls_id == i_label).nonzero()
                if indices.shape[0] <= n_view:
                    anc_features.append(features[i][indices].squeeze(1))
                else:
                    keep = torch.randperm(indices.shape[0])[:n_view]
                    indices = indices[keep]
                    anc_features.append(features[i][indices].squeeze(1))
                anc_labels.append(torch.full((indices.shape[0],), cls_id))

        anc_features = torch.cat(anc_features, dim=0)
        anc_labels = torch.cat(anc_labels).to(features.device)

        return anc_features, anc_labels

    def contrastive_loss(self, anchor, target):
        mask = torch.eq(target.unsqueeze(1), target.unsqueeze(0)).float().to(anchor.device)
        sim = torch.div(torch.matmul(anchor, anchor.T), 0.07)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        neg_mask = 1 - mask

        logits_mask = torch.ones((mask.shape[0], mask.shape[0]), dtype=bool).to(anchor.device)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(dim=1, keepdim=True)
        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-5)
        loss = -mean_log_prob_pos.mean()

        return loss

    def forward(self, features_embedding_all, label, text_embedding_all, ignore, sample_num = 10):
        loss = 0.0
        n_view_list = [5, 2, 1]
        n_view_list = [int(x * sample_num) for x in n_view_list]

        for features, text_embedding, n_view in zip(features_embedding_all, text_embedding_all, n_view_list):
            i_label = label.unsqueeze(1).float()
            i_label = F.interpolate(i_label, (features.shape[2], features.shape[3]), mode='nearest')
            i_label = i_label.long()
            assert i_label.shape[-1] == features.shape[-1], '{} {}'.format(i_label.shape, features.shape)

            batch_size = features.shape[0]
            i_label = i_label.view(batch_size, -1)
            features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, features.shape[1])

            anchor, target = self.sample_anchor(features, i_label, ignore, n_view)
            anchor = torch.cat([anchor, text_embedding], dim=0)
            anchor = F.normalize(anchor, p=2, dim=1)
            text_embedding_label = torch.tensor([0, 1, 2, 3]).repeat(self.prompt_num).to(target.device)
            target = torch.cat([target, text_embedding_label], dim=0)
            loss += self.contrastive_loss(anchor, target)

        return loss

class SoftmaxWeightedLoss(nn.Module):
    def __init__(self, num_cls=4):
        super(SoftmaxWeightedLoss, self).__init__()
        self.num_cls = num_cls

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_cls):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, output, target):
        target = target.float()
        B, _, H, W = output.size()
        target = self._one_hot_encoder(target)
        for i in range(self.num_cls):
            outputi = output[:, i, :, :]
            targeti = target[:, i, :, :]
            weighted = 1.0 - (torch.sum(targeti, (1, 2)) * 1.0 / torch.sum(target, (1, 2, 3)))
            weighted = torch.reshape(weighted, (-1, 1, 1)).repeat(1, H, W)
            if i == 0:
                cross_loss = 0
            else:
                cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        cross_loss = torch.mean(cross_loss)
        return cross_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - dice
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
            weight[0] = 0.5
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            loss_label = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - loss_label.item())
            loss += loss_label * weight[i]
        print(class_wise_dice)
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    
    elif pred.sum() < 200 and gt.sum() <= 200:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, image1, image2, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    image1, image2 = image1.squeeze(0).cpu().detach().numpy(), image2.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:  
        prediction = np.zeros_like(label)

        for ind in range(image.shape[2]):
            slice = image[:, :, ind]
            slice1 = image1[:, :, ind]
            slice2 = image2[:, :, ind]
            x, y = slice.shape[0], slice.shape[1]

            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  
                slice1 = zoom(slice1, (patch_size[0] / x, patch_size[1] / y), order=3)
                slice2 = zoom(slice2, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()  
            input1 = torch.from_numpy(slice1).unsqueeze(0).unsqueeze(0).float().cuda()
            input2 = torch.from_numpy(slice2).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input,input1,input2)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)  
                out = out.cpu().detach().numpy()

                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0) 
                else:
                    pred = out
                prediction[:, :, ind] = pred  
    else: 
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        input1 = torch.from_numpy(image1).unsqueeze(0).unsqueeze(0).float().cuda()
        input2 = torch.from_numpy(image2).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input,input1,input2), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    
    if test_save_path is not None:
            img_itk = sitk.GetImageFromArray(image.transpose(2,0,1).astype(np.float32))
            prd_itk = sitk.GetImageFromArray(prediction.transpose(2,0,1).astype(np.float32))
            lab_itk = sitk.GetImageFromArray(label.transpose(2,0,1).astype(np.float32))
            img_itk.SetSpacing((z_spacing, 1, 1))
            prd_itk.SetSpacing((z_spacing, 1, 1))
            lab_itk.SetSpacing((z_spacing, 1, 1))
            sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
            sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
            sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        if i == 3:
            prediction[prediction == 2] = 3
            label[label == 2] = 3
            metric_list.append(calculate_metric_percase(prediction == 3, label == 3))
            prediction[prediction >= 1] = 1
            label[label >= 1] = 1
            metric_list.append(calculate_metric_percase(prediction == 1, label == 1))    
    
    return metric_list
