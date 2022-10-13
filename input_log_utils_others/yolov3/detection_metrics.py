# -*- coding: utf-8 -*-
"""Reference.

https://pypi.org/project/mean-average-precision/#:~:text=mAP%3A%20Mean%20Average%20Precision%20for,truth%20and%20set%20of%20classes.
"""


import numpy as np
from mean_average_precision import MeanAveragePrecision2d
from mean_average_precision.utils import row_to_vars, check_box
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt


def zfill_array_with_zeros(arr, width):
    """
    Return array padded with zeros as necessary.

    Parameters
    ----------
    arr : np.array
        array to zfill.
    width : int
        desired width of last dimension.

    Returns
    -------
    out : np.array
        array with zeros padded.

    """
    out = np.zeros(shape=(arr.shape[0], width),
                   dtype=arr.dtype
                   )
    out[:, :arr.shape[-1]] = arr
    return out


def create_metric_object(list_of_gt_arrays,
                         list_of_predictions,
                         num_classes,
                         list_of_image_names=None
                         ):
    """
    Create the metric object that will be used for further calculations.

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int
        number of classes.
    list_of_image_names : List[str], optional
        a list of image names. Uses numbers in order if not provided.
        The default is None.

    Returns
    -------
    metric_obj : object of Class MeanAveragePrecision
        an instance of the MeanAveragePrecision class, containing our metrics.

    """
    assert len(list_of_gt_arrays) == len(list_of_predictions)

    metric_obj = MeanAveragePrecision2d(num_classes=num_classes)

    if list_of_image_names is None:
        list_of_image_names = list(range(len(list_of_gt_arrays)))

    image_to_name_mapping = {}
    for gt, preds, name in zip(list_of_gt_arrays,
                               list_of_predictions,
                               list_of_image_names):
        image_to_name_mapping[metric_obj.imgs_counter] = name
        assert gt.ndim == 2
        # assert preds.ndim == 2
        # pad zeros to ground truth for desired shape
        if gt.shape[-1] < 7:
            gt = zfill_array_with_zeros(gt, 7)

        # pad zeros to predictions for desired shape
        if preds.shape[-1] < 6:
            preds = zfill_array_with_zeros(preds, 6)
        # add preds and gt for each image to metric object
        metric_obj.add(preds, gt)
    for table in metric_obj.match_table:
        table['img_id'] = table['img_id'].map(image_to_name_mapping)

    return metric_obj


def get_mAP(list_of_gt_arrays,
            list_of_predictions,
            num_classes=1,
            IOU=[0.5, 0.75, 0.9],
            list_of_image_names=None):
    """
    Get mAP values for various IOU Thresholds.

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int, optional
        number of classes. The default is 1.
    IOU : List[floats], optional
        list of IOU thresholds. The default is [0.5, 0.75, 0.9].
    list_of_image_names : List[str], optional
        a list of image names. Uses numbers in order if not provided.
        The default is None.

    Returns
    -------
    mAP : Dict
        dictionary of mAP values for given IOU values.

    """
    metric_obj = create_metric_object(list_of_gt_arrays,
                                      list_of_predictions,
                                      num_classes,
                                      list_of_image_names
                                      )
    # OLD METHOD - calculate only at recall thresholds
    # out = metric_obj.value(iou_thresholds=IOU, recall_thresholds=np.arange(0., 1.1, 0.1))
    # NEW METHOD - Calculate across entire precision-recall curve.
    out = metric_obj.value(iou_thresholds=IOU)
    mAP = {threshold: out[threshold][0]['ap'] for threshold in IOU}
    return mAP


def get_tpr_fpr_rate(list_of_gt_arrays,
                     list_of_predictions,
                     num_classes=1,
                     IOU=np.arange(0, 1.05, 0.05),
                     mpolicy="greedy"):
    """
    Evaluate the tp and fp for various IOU thresholds.

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int, optional
        number of classes. The default is 1.
    IOU : List[floats], optional
        list of IOU thresholds. The default is np.arange(0, 1.05, 0.05).
    mpolicy : str, optional
        box matching policy.
        greedy - greedy matching like VOC PASCAL.
        soft - soft matching like COCO.
        The default is "greedy".

    Returns
    -------
    out : dict
        dictionary containing tp and fp for iou thresholds.

    """
    metric_obj = create_metric_object(list_of_gt_arrays,
                                      list_of_predictions,
                                      num_classes)
    metric = {}
    for t in IOU:
        metric[t] = {}
        for class_id in range(num_classes):
            tp, fp = get_tp_fp(metric_obj, class_id, t, mpolicy)
            metric[t][class_id] = {}
            metric[t][class_id]["tp"] = tp
            metric[t][class_id]["fp"] = fp
    return metric


def get_fdr(list_of_gt_arrays,
            list_of_predictions,
            num_classes=1,
            IOU=[0.5, 0.75, 0.9],
            mpolicy="greedy"):
    """
    Evaluate FDR (false discovery rate) for various IOU thresholds.

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int, optional
        number of classes. The default is 1.
    IOU : List[floats], optional
        list of IOU thresholds. The default is [0.5, 0.75, 0.9].
    mpolicy : str, optional
        box matching policy.
        greedy - greedy matching like VOC PASCAL.
        soft - soft matching like COCO.
        The default is "greedy".

    Returns
    -------
    out : dict
        dictionary containing tp and fp for iou thresholds.

    """
    metric_obj = create_metric_object(list_of_gt_arrays,
                                      list_of_predictions,
                                      num_classes)
    metric = {k:{} for k in range(num_classes)}
    for t in IOU:
        for class_id in range(num_classes):
            tp, fp = get_tp_fp(metric_obj, class_id, t, mpolicy)
            metric[class_id][t] = sum(tp)/len(tp)
    return metric


def get_tp_fp(self, class_id, iou_threshold, mpolicy="greedy"):
    """
    Evaluate class.

    Parameters
    ----------
    class_id : int
        index of evaluated class.
    iou_threshold : float
        DESCRIPTION.
    mpolicy : str, optional
        box matching policy.
        greedy - greedy matching like VOC PASCAL.
        soft - soft matching like COCO.
        The default is "greedy".

    Returns
    -------
    tp : np.array
        array containing tp.
    fp : np.array
        array containing fp.

    """
    table = self.match_table[class_id].sort_values(by=['confidence'], ascending=False)
    matched_ind = {}
    nd = len(table)
    tp = np.zeros(nd, dtype=np.float64)
    fp = np.zeros(nd, dtype=np.float64)
    for d in range(nd):
        img_id, conf, iou, difficult, crowd, order = row_to_vars(table.iloc[d])
        if img_id not in matched_ind:
            matched_ind[img_id] = []
        res, idx = check_box(
            iou,
            difficult,
            crowd,
            order,
            matched_ind[img_id],
            iou_threshold,
            mpolicy
        )
        if res == 'tp':
            tp[d] = 1
            matched_ind[img_id].append(idx)
        elif res == 'fp':
            fp[d] = 1

    return tp, fp


def custom_page_accuracy_metric(df, is_strict=False):

    """
    Calculate accuracy in a df of images.

    Page-level accuracy is defined as
    ---
    how many pages out of the total had all of the signatures
    correctly detected
    If is_strict is True, then no false positives are allowed.
    ---

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing image level counts
        of ground truth and true positives.
    is_strict : TYPE, optional
        indicates whether false positives are allowed or not.
        When strict, false positives are not allowed.
        The default is False.

    Returns
    -------
    output : dict
        dictionary containing accuracy and total counts.

    """
    output = {}
    if is_strict:
        totals = (
                  (df['tp_per_image'] == df['gt_counts'])
                  &
                  (df['fp_per_image'] == 0)
                 ).sum()
    else:
        totals = (df['tp_per_image'] == df['gt_counts']).sum()
    output["custom_accuracy"] = totals / len(df)
    output["counts"] = len(df)
    return output


def get_page_level_accuracy(list_of_gt_arrays,
                            list_of_predictions,
                            num_classes=1,
                            IOU=[0.5, 0.75, 0.9],
                            mpolicy="greedy"):
    """
    Evaluate page level accuracy for a "dataset level" accuracy metric.

    Page-level accuracy is defined as
    ---
    how many pages out of the total had all of the signatures
    correctly detected
    ---
    Page-level accuracy @ different numbers of signatures per page

    how many pages with 1 signature have all signatures correctly detected
    how many pages with 2 signatures have all signatures correctly detected
    how many pages with 3 signatures have all signatures correctly detected
    etc.

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int, optional
        number of classes. The default is 1.
    IOU : List[floats], optional
        list of IOU thresholds. The default is [0.5, 0.75, 0.9].
    mpolicy : str, optional
        box matching policy.
        greedy - greedy matching like VOC PASCAL.
        soft - soft matching like COCO.
        The default is "greedy".

    Returns
    -------
    metric : dict
        evaluated page level accuracy.

    """
    gt_counts = np.array([len(arr) for arr in list_of_gt_arrays])
    metric_obj = create_metric_object(list_of_gt_arrays,
                                      list_of_predictions,
                                      num_classes)
    metric = {}
    for t in IOU:
        metric[t] = {}
        for class_id in range(num_classes):
            tp, fp = get_tp_fp(metric_obj, class_id, t, mpolicy)
            table = metric_obj.match_table[class_id].sort_values(by=['confidence'], ascending=False)
            table['tp'] = tp
            table['fp'] = fp
            images_df = table.groupby('img_id')['tp', 'fp'].sum()
            # Handle Case where no boxes were predicted
            # But ground truth contained a box
            if len(images_df) < len(gt_counts):
                corrected_tp = np.zeros_like(gt_counts)
                corrected_fp = np.zeros_like(gt_counts)
                img_indexes = [len(preds) > 0 for preds in list_of_predictions]
                corrected_tp[img_indexes] = images_df['tp']
                corrected_fp[img_indexes] = images_df['fp']
                index = np.arange(1, len(corrected_tp) + 1)
                images_df = pd.DataFrame({"tp": corrected_tp, "fp": corrected_fp},
                                         index=index)
            temp = pd.DataFrame({"image_id": images_df.index,
                                 "tp_per_image": images_df['tp'],
                                 "fp_per_image": images_df['fp'],
                                 "gt_counts": gt_counts
                                 })
            page_accuracy = custom_page_accuracy_metric(temp)
            page_accuracy_for_signature_counts = temp.groupby("gt_counts").apply(custom_page_accuracy_metric).to_dict()
            metric[t][class_id] = {}
            metric[t][class_id]["page_accuracy"] = page_accuracy
            metric[t][class_id]["page_accuracy_for_signature_counts"] = page_accuracy_for_signature_counts
    return metric


def get_page_level_strict_accuracy(list_of_gt_arrays,
                                   list_of_predictions,
                                   num_classes=1,
                                   IOU=[0.5, 0.75, 0.9],
                                   mpolicy="greedy"):
    """
    Evaluate "Strict" page level accuracy for a "dataset level" accuracy metric.

    Strict Page-level accuracy is defined as
    ---
    how many pages out of the total had all of the signatures
    correctly detected and no false positives.
    ---
    Page-level accuracy @ different numbers of signatures per page

    how many pages with 1 signature have all signatures correctly detected
    with no false positives
    how many pages with 2 signatures have all signatures correctly detected
    with no false positives
    how many pages with 3 signatures have all signatures correctly detected
    with no false positives
    etc.

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int, optional
        number of classes. The default is 1.
    IOU : List[floats], optional
        list of IOU thresholds. The default is [0.5, 0.75, 0.9].
    mpolicy : str, optional
        box matching policy.
        greedy - greedy matching like VOC PASCAL.
        soft - soft matching like COCO.
        The default is "greedy".

    Returns
    -------
    metric : dict
        evaluated page level strict accuracy.

    """
    gt_counts = np.array([len(arr) for arr in list_of_gt_arrays])
    metric_obj = create_metric_object(list_of_gt_arrays,
                                      list_of_predictions,
                                      num_classes)
    metric = {}
    for t in IOU:
        metric[t] = {}
        for class_id in range(num_classes):
            tp, fp = get_tp_fp(metric_obj, class_id, t, mpolicy)
            table = metric_obj.match_table[class_id].sort_values(by=['confidence'], ascending=False)
            table['tp'] = tp
            table['fp'] = fp
            images_df = table.groupby('img_id')['tp', 'fp'].sum()
            # Handle Case where no boxes were predicted
            # But ground truth contained a box
            if len(images_df) < len(gt_counts):
                corrected_tp = np.zeros_like(gt_counts)
                corrected_fp = np.zeros_like(gt_counts)
                img_indexes = [len(preds) > 0 for preds in list_of_predictions]
                corrected_tp[img_indexes] = images_df['tp']
                corrected_fp[img_indexes] = images_df['fp']
                index = np.arange(1, len(corrected_tp) + 1)
                images_df = pd.DataFrame({"tp": corrected_tp, "fp": corrected_fp},
                                         index=index)
            temp = pd.DataFrame({"image_id": images_df.index,
                                 "tp_per_image": images_df['tp'],
                                 "fp_per_image": images_df['fp'],
                                 "gt_counts": gt_counts
                                 })
            page_accuracy = custom_page_accuracy_metric(temp, is_strict=True)
            page_accuracy_for_signature_counts = temp.groupby("gt_counts").apply(custom_page_accuracy_metric, is_strict=True).to_dict()
            metric[t][class_id] = {}
            metric[t][class_id]["page_accuracy"] = page_accuracy
            metric[t][class_id]["page_accuracy_for_signature_counts"] = page_accuracy_for_signature_counts
    return metric



def get_custom_tpr_fpr_rate(list_of_gt_arrays,
                            list_of_predictions,
                            num_classes=1,
                            IOU=np.arange(0, 1.05, 0.05),
                            mpolicy="greedy"):
    """
    Evaluate custom tpr and fpr for a tpr vs fpr metric.

    We avoid using TN for our calculations as it is ill-defined in object
    detection models.
    As a consequence, we do not have a real FPR, but a proxy for FPR
    that is evaluated as FP / (TP + FN)
    i.e., it is the rate of FP against ground truth images

    Parameters
    ----------
    list_of_gt_arrays : List[np.array]
        a list of arrays for ground truth boxes, each array is 2d
        representing ground truth boxes per image.
    list_of_predictions : List[np.array]
        a list of arrays, each array is 2d
        representing the predictions per image.
    num_classes : int, optional
        number of classes. The default is 1.
    IOU : List[floats], optional
        list of IOU thresholds. The default is np.arange(0, 1.05, 0.05).
    mpolicy : str, optional
        box matching policy.
        greedy - greedy matching like VOC PASCAL.
        soft - soft matching like COCO.
        The default is "greedy".

    Returns
    -------
    rate_metric : dict
        evaluated tpr_fpr_metrics.

    """
    gt_counts = np.array([len(arr) for arr in list_of_gt_arrays])
    metric_obj = create_metric_object(list_of_gt_arrays,
                                      list_of_predictions,
                                      num_classes)

    rate_metric = {k: defaultdict(list) for k in range(num_classes)}
    for t in IOU:
        for class_id in range(num_classes):
            tp, fp = get_tp_fp(metric_obj, class_id, t, mpolicy)
            tpr = tp.sum() / gt_counts.sum()
            fpr_proxy = min(fp.sum() / gt_counts.sum(), 1)
            rate_metric[class_id]["tpr"].append(tpr)
            rate_metric[class_id]["fpr_proxy"].append(fpr_proxy)
    rate_metric['IOU'] = IOU
    return rate_metric


def plot_rate_metric(rate_metric, class_id=0):
    """
    Plot tpr and fpr proxy against iou for a given class.

    Parameters
    ----------
    rate_metric : dict
        rate_metric dict containing tpr and fpr proxy values.
    class_id : int, optional
        class id for which the values are plotted. The default is 0.

    Returns
    -------
    None.

    """
    plt.plot(rate_metric['IOU'],
             rate_metric[class_id]['tpr'],
             label='tpr')
    plt.plot(rate_metric['IOU'],
             rate_metric[class_id]['fpr_proxy'],
             label='fpr_proxy')
    plt.xlabel('IOU Thresholds')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    # gt = np.array([
    #     [439, 157, 556, 241, 0, 0, 0],
    # ])

    gt = np.array([
        [439, 157, 556, 241],
        [580, 250, 640, 350],
        [220, 250, 330, 350]
    ])

    # [xmin, ymin, xmax, ymax, class_id, confidence]
    preds = np.array([
        [432, 153, 551, 220, 0, 0],
        [439, 157, 556, 241, 0, 0],
    ])

    list_of_gt_arrays = [gt] * 1
    list_of_predictions = [preds] * 1
    list_of_image_names = list("a")

    mAP = get_mAP(list_of_gt_arrays, list_of_predictions, list_of_image_names=list_of_image_names)
    rate_metric = get_custom_tpr_fpr_rate(list_of_gt_arrays, list_of_predictions)
    page_accuracy = get_page_level_accuracy(list_of_gt_arrays, list_of_predictions)
    fdr = get_fdr(list_of_gt_arrays, list_of_predictions)
    print(mAP)
    print(rate_metric)
    plot_rate_metric(rate_metric)
