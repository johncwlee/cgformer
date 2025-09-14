import numpy as np

class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)
    
    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum
    
    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (self.completion_tp + self.completion_fp + self.completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)

        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }
        
    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, n)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, n)
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :] # ground truth
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum
    
    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]
        _C = self.n_classes

        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, n)
        predict = predict.reshape(_bs, -1)  # (_bs, n)

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]
            y_pred = predict[idx, :]

            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
                y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            
            for j in range(_C):
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn
        
        return tp_sum, fp_sum, fn_sum


class GroupedSSCMetrics(SSCMetrics):
    def __init__(self, n_classes, rare_classes):
        self.rare_class_indices = rare_classes
        assert rare_classes is not None and len(rare_classes) > 0, "Rare classes must be specified"

        # Number of classes after grouping (all rare merged into one)
        num_classes = n_classes - len(rare_classes) + 1
        super(GroupedSSCMetrics, self).__init__(num_classes)

        # Build a mapping from original class ids -> grouped contiguous ids
        # Keep the largest rare index as the representative in order, but remap
        # to contiguous indices [0, num_classes-1] for metric computation.
        self.original_num_classes = n_classes
        self.agg_old_index = int(max(self.rare_class_indices))

        kept_indices = [i for i in range(n_classes)
                        if (i not in self.rare_class_indices) or (i == self.agg_old_index)]
        # kept_indices are in ascending order; the representative rare class will naturally
        # become the last index if it's the max of the rare set.
        self.old_to_new = np.zeros(n_classes, dtype=np.int32)
        for new_idx, old_idx in enumerate(kept_indices):
            self.old_to_new[old_idx] = new_idx
        # Map all rare classes to the representative new index
        self.agg_new_index = int(self.old_to_new[self.agg_old_index])
        for r in self.rare_class_indices:
            self.old_to_new[r] = self.agg_new_index

    def _remap_labels(self, labels: np.ndarray) -> np.ndarray:
        # Preserve ignore index 255
        mapped = np.copy(labels)
        valid = mapped != 255
        # Ensure integer type for indexing
        mapped = mapped.astype(np.int32, copy=False)
        mapped[valid] = self.old_to_new[mapped[valid]]
        return mapped
    
    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1

        #? Remap predictions and GT to grouped class space
        y_pred_g = self._remap_labels(y_pred)
        y_true_g = self._remap_labels(y_true)

        mask = y_true_g != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface

        tp, fp, fn = self.get_score_completion(y_pred_g, y_true_g, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true_g != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred_g, y_true_g, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

class SemanticSegmentationMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        # For semantic segmentation metrics (per-class)
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)
    
    def add_batch(self, y_pred, y_true):
        """
        Args:
            y_pred: Predictions of shape (B, num_classes, H, W) as logits or probabilities,
                    or (B, H, W) as class indices.
            y_true: Ground truth of shape (B, 1, H, W) or (B, H, W) as class indices.
        """
        # Convert predictions from (B, C, H, W) to (B, H, W) class indices
        if len(y_pred.shape) == 4 and y_pred.shape[1] == self.n_classes:
            y_pred = np.argmax(y_pred, axis=1)  # (B, H, W)
        
        # Remove channel dimension from ground truth if present
        if len(y_true.shape) == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)  # (B, H, W)
        
        # Create base mask for valid pixels (ignore class 255)
        mask = y_true != 255

        # Compute per-class semantic segmentation metrics
        tp, fp, fn = self.get_score_semantic(
            y_pred, y_true, mask
        )
        self.tps += tp
        self.fps += fp
        self.fns += fn
    
    def get_stats(self):
        """Calculate and return all metrics"""
        # Compute per-class IoU
        iou_class = self.tps / (self.tps + self.fps + self.fns + 1e-5)

        return {
            "iou_class": iou_class,
            "iou_mean": np.mean(iou_class[1:]),  # Exclude background class (class 0)
        }
    
    def get_score_semantic(self, predict, target, mask):
        """Compute per-class semantic segmentation metrics in a vectorized way."""
        _C = self.n_classes

        # Apply mask and flatten
        y_pred = predict[mask]
        y_true = target[mask]

        tp = np.zeros(_C, dtype=np.int32)
        fp = np.zeros(_C, dtype=np.int32)
        fn = np.zeros(_C, dtype=np.int32)

        # Compute TP, FP, FN for each class
        for j in range(_C):
            tp[j] = np.sum(np.logical_and(y_true == j, y_pred == j))
            fp[j] = np.sum(np.logical_and(y_true != j, y_pred == j))
            fn[j] = np.sum(np.logical_and(y_true == j, y_pred != j))
        
        return tp, fp, fn

class FOVSSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)
    
    def add_batch(self, y_pred, y_true, 
                  fov_mask=None, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if fov_mask is not None:
            mask = mask & fov_mask
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum
    
    def get_stats(self):
        if self.completion_tp != 0:
            iou = self.completion_tp / (self.completion_tp + self.completion_fp + self.completion_fn)
        else:
            iou = 0
        
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)

        return {
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }
        
    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, n)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, n)
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :] # ground truth
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum
    
    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]
        _C = self.n_classes

        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, n)
        predict = predict.reshape(_bs, -1)  # (_bs, n)

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]
            y_pred = predict[idx, :]

            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
                y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            
            for j in range(_C):
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn
        
        return tp_sum, fp_sum, fn_sum