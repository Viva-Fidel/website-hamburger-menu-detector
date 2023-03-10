import torch
import cv2


class Detection:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, img, model):
        self.img = img
        self.model = model

    def do_detection(self):
        results = self.model(self.img)  # receiving results
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        x_shape, y_shape = self.img.shape[1], self.img.shape[0]
        n = len(labels)

        for i in range(n):
            row = coordinates[i]

            if row[4] >= 0.50:
                x1, y1, x2, y2 = int(row[0] * x_shape), int((row[1] * y_shape)), int(
                    row[2] * x_shape), int((row[3] * y_shape))

                cv2.rectangle(self.img, (x1-10, y1-10), (x2+10, y2+10), (255, 0, 0), 2)
                return self.img
            else:
                return self.img
