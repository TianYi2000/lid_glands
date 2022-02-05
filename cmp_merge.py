import numpy as np
import os
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont

ttfont = ImageFont.truetype("JetBrainsMono-Regular.ttf",30)
target_h, target_w = 420, 890
eps = 1.0e-6
def cal(ref_img, out_img):
    TP = out_img & ref_img
    TN = ~out_img & ~ref_img
    FP = out_img & ~ref_img
    FN = ~out_img & ref_img

    TP, TN, FP, FN = [idx.sum() for idx in (TP, TN, FP, FN)]
    IoU = (TP / (TP + FP + FN + eps))
    precision = (TP / (TP + FP + eps))
    recall = (TP / (TP + FN + eps))
    F1 = (2 * (TP / (TP + FP + eps)) * (TP / (TP + FN + eps)) / ((TP / (TP + FP + eps)) + (TP / (TP + FN + eps)) + eps))
    sensiticity = (TP / (TP + FN + eps))
    specificity = (TN / (TN + FP + eps))
    return IoU, precision, recall, F1, sensiticity, specificity

# def test_and_merge(img_type, epoch):
#     assert img_type in ['area', 'gland']
#
#     origin_path = os.path.join('data/eyelid/train_up/val_' + img_type, 'img')
#     pred_path = os.path.join('output/' + img_type, f'epoch-{epoch}/' + 'pred')
#     label_path = os.path.join('data/eyelid/train_up/val_' + img_type, 'labelcol')
#
#     saved_path = os.path.join('output/merge/', img_type)
#     os.makedirs(saved_path,exist_ok=True)
#
#     file_list = os.listdir(pred_path)
#
#     IoU, precision, recall, sensiticity, specificity, F1 = [], [], [], [], [], []
#
#     for file in tqdm(file_list, ncols=100):
#         origin_img = Image.open(os.path.join(origin_path, file)).convert('RGB')
#         label_img = Image.open(os.path.join(label_path, file)).convert('1')
#         pred_img = Image.open(os.path.join(pred_path, file)).convert('1').resize((target_w, target_h), Image.BICUBIC)
#
#         imagefile = [origin_img, label_img, pred_img]
#         npfile = [np.asarray(label_img), np.asarray(pred_img)]
#
#         now_IoU, now_precision, now_recall, now_F1, now_sensiticity, now_specificity = cal(npfile[0], npfile[1])
#
#         IoU.append(now_IoU)
#         precision.append(now_precision)
#         recall.append(now_recall)
#         F1.append(now_F1)
#         sensiticity.append(now_sensiticity)
#         specificity.append(now_specificity)
#
#         target = Image.new('RGB', (target_w, target_h * 4))
#         left = 0
#         right = target_h
#         for image in imagefile:
#             target.paste(image, (0, left, target_w, right))# 将image复制到target的指定位置中
#             left += target_h
#             right += target_h
#             quality_value = 100
#
#             draw = ImageDraw.Draw(target)
#             draw.text((10,left),f'IoU={now_IoU}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 50),f'precision={now_precision}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 100),f'recall={now_recall}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 150),f'F1={now_F1}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 200),f'sensiticity={now_sensiticity}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 250),f'specificity={now_specificity}', fill=(255,255,255),font=ttfont)
#             target.save(os.path.join(saved_path, file), quality = quality_value)
#     print(f'IoU: {np.mean(IoU):.4%}, precision: {np.mean(precision):.4%}, recall: {np.mean(recall):.4%},\nF1 score: {np.mean(F1):.4%}, sensiticity: {np.mean(sensiticity):.4%}, specificity: {np.mean(specificity):.4%}')

# def select_both():
#     epoch =
#     {
#         'area':60,
#         'gland':50
#     }
#     origin_path, pred_path, label_path = []
#
#     for img_type in ['area', 'gland']:
#
#         origin_path .append( os.path.join('data/eyelid/train/val_' + img_type, 'img') )
#         pred_path .append( os.path.join('output/' + img_type, f'epoch-{epoch}/' + 'pred') )
#         label_path .append( os.path.join('data/eyelid/train/val_' + img_type, 'labelcol') )
#
#     saved_path = os.path.join('output/merge/', img_type)
#     os.makedirs(saved_path,exist_ok=True)
#
#     file_list = os.listdir(pred_path)
#
#     IoU, precision, recall, sensiticity, specificity, F1 = [], [], [], [], [], []
#
#     for file in tqdm(file_list, ncols=100):
#         origin_img = Image.open(os.path.join(origin_path, file)).convert('RGB')
#         label_img = Image.open(os.path.join(label_path, file)).convert('1')
#         pred_img = Image.open(os.path.join(pred_path, file)).convert('1').resize((target_w, target_h), Image.BICUBIC)
#
#         imagefile = [origin_img, label_img, pred_img]
#         npfile = [np.asarray(label_img), np.asarray(pred_img)]
#
#         now_IoU, now_precision, now_recall, now_F1, now_sensiticity, now_specificity = cal(npfile[0], npfile[1])
#
#         IoU.append(now_IoU)
#         precision.append(now_precision)
#         recall.append(now_recall)
#         F1.append(now_F1)
#         sensiticity.append(now_sensiticity)
#         specificity.append(now_specificity)
#
#         target = Image.new('RGB', (target_w, target_h * 4))
#         left = 0
#         right = target_h
#         for image in imagefile:
#             target.paste(image, (0, left, target_w, right))# 将image复制到target的指定位置中
#             left += target_h
#             right += target_h
#             quality_value = 100
#
#             draw = ImageDraw.Draw(target)
#             draw.text((10,left),f'IoU={now_IoU}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 50),f'precision={now_precision}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 100),f'recall={now_recall}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 150),f'F1={now_F1}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 200),f'sensiticity={now_sensiticity}', fill=(255,255,255),font=ttfont)
#             draw.text((10,left + 250),f'specificity={now_specificity}', fill=(255,255,255),font=ttfont)
#             target.save(os.path.join(saved_path, file), quality = quality_value)
#     print(f'IoU: {np.mean(IoU):.4%}, precision: {np.mean(precision):.4%}, recall: {np.mean(recall):.4%},\nF1 score: '
#           f'{np.mean(F1):.4%}, sensiticity: {np.mean(sensiticity):.4%}, specificity: {np.mean(specificity):.4%}')


def test_and_merge(img_type):
    assert img_type in ['area', 'gland']

    origin_path = os.path.join('data/eyelid/train/val_' + img_type, 'img')
    pred_path = os.path.join('output/' + img_type + '_up_and_bottom', 'pred')
    label_path = os.path.join('data/eyelid/train/val_' + img_type, 'labelcol')

    saved_path = os.path.join('output/merge/', img_type)
    os.makedirs(saved_path,exist_ok=True)

    file_list = os.listdir(pred_path)

    IoU, precision, recall, sensiticity, specificity, F1 = [], [], [], [], [], []

    for file in tqdm(file_list, ncols=100):
        origin_img = Image.open(os.path.join(origin_path, file)).convert('RGB')
        label_img = Image.open(os.path.join(label_path, file)).convert('1')
        pred_img = Image.open(os.path.join(pred_path, file)).convert('1').resize((target_w, target_h), Image.BICUBIC)

        imagefile = [origin_img, label_img, pred_img]
        npfile = [np.asarray(label_img), np.asarray(pred_img)]

        now_IoU, now_precision, now_recall, now_F1, now_sensiticity, now_specificity = cal(npfile[0], npfile[1])

        IoU.append(now_IoU)
        precision.append(now_precision)
        recall.append(now_recall)
        F1.append(now_F1)
        sensiticity.append(now_sensiticity)
        specificity.append(now_specificity)

        target = Image.new('RGB', (target_w, target_h * 4))
        left = 0
        right = target_h
        for image in imagefile:
            target.paste(image, (0, left, target_w, right))# 将image复制到target的指定位置中
            left += target_h
            right += target_h
            quality_value = 100

            draw = ImageDraw.Draw(target)
            draw.text((10,left),f'IoU={now_IoU}', fill=(255,255,255),font=ttfont)
            draw.text((10,left + 50),f'precision={now_precision}', fill=(255,255,255),font=ttfont)
            draw.text((10,left + 100),f'recall={now_recall}', fill=(255,255,255),font=ttfont)
            draw.text((10,left + 150),f'F1={now_F1}', fill=(255,255,255),font=ttfont)
            draw.text((10,left + 200),f'sensiticity={now_sensiticity}', fill=(255,255,255),font=ttfont)
            draw.text((10,left + 250),f'specificity={now_specificity}', fill=(255,255,255),font=ttfont)
            target.save(os.path.join(saved_path, file), quality = quality_value)
    print(f'IoU: {np.mean(IoU):.4%}, precision: {np.mean(precision):.4%}, recall: {np.mean(recall):.4%},\nF1 score: {np.mean(F1):.4%}, sensiticity: {np.mean(sensiticity):.4%}, specificity: {np.mean(specificity):.4%}')

# for epoch in range(10, 60, 10):
#     print('epoch=', epoch)
#     print('area')
#     test_and_merge('area', epoch)
#     print('gland')
#     test_and_merge('gland', epoch)
# test_and_merge('area', 60)
# test_and_merge('gland', 50)

test_and_merge('area')
test_and_merge('gland')

