import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, recall_score,f1_score
#多分类和二分类混淆矩阵和各项指标的计算公式各有不同
class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):


        tp = np.zeros(self.num_class, dtype=int)
        fp = np.zeros(self.num_class, dtype=int)
        tn = np.zeros(self.num_class, dtype=int)
        fn = np.zeros(self.num_class, dtype=int)

        for i in range(self.num_class):
            tp[i] = self.confusion_matrix[i, i]
            fp[i] = np.sum(self.confusion_matrix[:, i]) - tp[i]
            fn[i] = np.sum(self.confusion_matrix[i, :]) - tp[i]
            tn[i] = np.sum(self.confusion_matrix) - (tp[i] + fp[i] + fn[i])

        return tp, fp, tn, fn

    # def get_tp_fp_tn_fn(self):
    #     tp = np.diag(self.confusion_matrix)
    #     fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
    #     fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
    #     tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
    #     return tp, fp, tn, fn
    #
#准确率
    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision
#召回率
    def Recall(self):

        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall
#F1值
    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1
#整体准确率
    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA
#交并比
    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU
#平均交并比
    def Mean_Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        mIoU = np.mean(IoU)
        return mIoU
#Dice系数
    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

#Kappa系数
    def kappa(self):

        total_samples = np.sum(self.confusion_matrix)

        # Calculate the observed agreement (po)
        po = np.trace(self.confusion_matrix) / total_samples

        # Calculate the expected agreement (pe)
        row_marginals = np.sum(self.confusion_matrix, axis=1) / total_samples
        col_marginals = np.sum(self.confusion_matrix, axis=0) / total_samples
        pe = np.sum(row_marginals * col_marginals)

        # Calculate kappa
        kappa = (po - pe) / (1 - pe)

        return kappa

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


if __name__ == '__main__':

    gt = np.array([[0, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

    pre = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])


    eval = Evaluator(num_class=2)
    eval.add_batch(gt, pre)

    print("混淆矩阵\n",eval.confusion_matrix)

    # print(eval.get_tp_fp_tn_fn())
    print('公式计算的精度',eval.Precision())
    print('公式计算的召回率',eval.Recall())
    print('公式计算的交并比',eval.Intersection_over_Union())

    print('公式计算的F1值',eval.F1())
    print('公式计算的Kappa系数',eval.kappa())
    print('公式计算的mIoU',eval.Mean_Intersection_over_Union())
    print('公式计算的OA值',eval.OA())