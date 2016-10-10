""" 
The Detector Class holds several helpers to convert the output mask of the CNN into bounding box location 
It also holds the logic to compare bouding boxes and assess the detection outcome
"""
import numpy as np

class LogoDetector(object):

    def __init__(self, min_area=9, iou_threshold=0.25):
        """
        min_area: (pixels) min area to consider a detection successfull
        iou_threshold: Percentage of overlapping for the boxes predicted mask to be considered a true detection
        """
        self.iou_threshold = iou_threshold
        self.min_area = min_area
        self.max_distance = 2

    def get_distance(self, b_a, b_b):
        """ Compute the distance between 2 bounding boxes
        """
        dx = max(max(0, b_a[1] - b_b[3]), max(0, b_b[1] - b_a[3]))
        dy = max(max(0, b_a[0] - b_b[2]), max(0, b_b[0] - b_a[2]))
        return max(dx, dy)
    
    @staticmethod 
    def inter_over_union(A, B):
        """ Calculate the intersection over union from the given bounding boxes
        """
        inter = max(0, min(A[2], B[2]) - max(A[0], B[0])) * max(0, min(A[3], B[3]) - max(A[1], B[1]))
        a_area = (A[2] - A[0]) * (A[3] - A[1])
        b_area = (B[2] - B[0]) * (B[3] - B[1])
        union = a_area + b_area - inter
        return float(inter)/float(union)

    @staticmethod
    def merge_boxes(b_a, b_b):
        """ Merge the 2 bounding boxes in a new tuple
        """
        x1 = min(b_a[0], b_b[0])
        y1 = min(b_a[1], b_b[1])
        x2 = max(b_a[2], b_b[2])
        y2 = max(b_a[3], b_b[3])
        area = b_a[4]+1
        return (x1, y1, x2, y2, area)

    def eval_detections(self, detA, detB):
        a_o_u = LogoDetector.inter_over_union(detA, detB)
        if a_o_u > self.iou_threshold:
            return True
        return False
        
    def bounding_boxes(self, detections):
        """
        Find Bounding boxes from detections
        start with 1x1 pixel boxes and merge to find where the logo is
        """
        bboxes = []
        while len(detections) > 0:
            det = detections.pop(0)
            merging = True
            while merging:
                merging = False
                pointer = 0
                while pointer < len(detections):
                    if self.get_distance(det, detections[pointer]) <= self.max_distance:
                        det = self.merge_boxes(det, detections[pointer])
                        merging = True
                        detections.pop(pointer)
                    else:
                        pointer += 1
            if det[4] >= self.min_area:
                bboxes.append(det)
        return bboxes

    def detect(self, mask):
        """ Extract dection bounding boxes from mask
        """
        # 1) Return Non zero indices
        det_idx = np.where(mask > 0.0)
        idx_x, idx_y = det_idx[0], det_idx[1]
        # 2) Create 1x1 box for each pixel detected.
        detections = []
        for i in range(0, len(idx_x)):
            x, y = idx_x[i], idx_y[i]
            detections.append((x, y, x+1, y+1, 1)) # x1, y1, x2, y2, area
        # 3) merge boxes
        bounding_boxes = self.bounding_boxes(detections)
        return bounding_boxes

# Load Dataset and test Mask Extraction procedure
if __name__ == "__main__":
    from dataset import DataSet
    ds = DataSet("./debug", 180, 45, batch_size=2)
    detector = LogoDetector()
    for (dtrain, masks, labels) in ds.iterate_train_minibatch():
        for ii in range(0, dtrain.shape[0]):
            im = dtrain[ii, :, :, 0]
            mas = masks[ii, :, :, :]
            label = labels[ii]
            ma = np.argmax(mas, axis=2)
            print("Label {}".format(label))
            dt = detector.detect(ma)
            print(dt)


