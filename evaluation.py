from detecto_core import Model
from utils import show_labeled_image, read_image
import numpy as np
from time import time
from natsort import natsorted
import glob
import cv2


def evaluate(model, test, gt_labels, output_path,  OutOfSample=False):
    thresh = 0.6

    model = Model.load(model, gt_labels)

    test_data = natsorted(glob.glob(test + "/*.png"))

    for i, image_path in enumerate(test_data):
        if OutOfSample:
            img_id = image_path.split('out_of_sample/')[1].split('.png')[0]
        else:
            img_id = image_path.split('test_dir/')[1].split('.png')[0]
        image = read_image(image_path)
        start = time()
        labels, boxes, scores = model.predict(image)
        end = time()
        print("time to prediction:", end - start)

        # prediction_VS_planogram(labels, boxes, scores)

        filtered_indices = np.where(scores > thresh)
        filtered_scores = scores[filtered_indices]
        filtered_boxes = boxes[filtered_indices]
        num_list = filtered_indices[0].tolist()
        filtered_labels = [labels[i] for i in num_list]
        filtered_labels = map_barcode_to_labels(gt_labels, filtered_labels)
        show_labeled_image(image, filtered_boxes, filtered_labels, filtered_scores, img_id, output_path)


def predict_from_foscam(model, gt_labels, output_path):
    """
    The method preforms prediction on live frames from foscam camera
    :return: labels, bboxs, scores
    :rtype:labels- list of str, bboxes- Tensor of tensors, scores - Tensor

    """

    # Catch frame
    rtsp_url = 'rtsp://pi:pi123456!@199.203.102.124:86/videoSub'
    cap = cv2.VideoCapture(rtsp_url)
    frame_id = 0

    while (cap.isOpened()):

        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        if ret:
            start = time()
            frame_id += 1
            img_id = str(frame_id) + "_" + str(fps)

            #capture each 10th frame
            if frame_id % 1 == 0:
                print("frame number", frame_id)
                cv2.imshow('frame', frame)

                # pre-processing
                h, w = frame.shape[:2]
                new_h, new_w = h, w
                # cv2.imwrite("undistorted_fisheye_1.jpg", frame)

                # Define the fisheye model and distortion coefficients
                DIM = (new_w, new_h)
                K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
                D = np.array([-0.5, 0.5, 0.0, 0.0])

                # Perform the inverse fisheye correction
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
                undistorted_img = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
                cv2.imwrite("new_fix_image", undistorted_img)
                dtype = 'stereographic'
                format = 'fullframe'
                fov = 150
                pfov = 120
                img_out = f"./images/out/example3_{dtype}_{format}_{pfov}_{fov}.jpg"
                #obj = Defisheye(frame, dtype=dtype, format=format, fov=fov, pfov=pfov)
                #out =obj.convert(img_out)
                # Save the undistorted image
                out = cv2.resize(out, (new_w, new_h))

                cv2.imwrite("undistorted_fisheye_"+dtype+"_pov_"+str(fov)+"_pfov_" + str(pfov) +".jpg", out)
                # cv2.imwrite("with_fisheye_1.jpg", frame)

                # cv2.imshow("undistorted_fisheye_1", out)

                #Evaluation
                thresh = 0.6
                model = Model.load(model, gt_labels)
                start = time()
                labels, boxes, scores = model.predict(frame)
                end = time()
                print("time to prediction:", end - start)
                filtered_indices = np.where(scores > thresh)
                filtered_scores = scores[filtered_indices]
                filtered_boxes = boxes[filtered_indices]
                num_list = filtered_indices[0].tolist()
                filtered_labels = [labels[i] for i in num_list]
                filtered_labels = map_barcode_to_labels(filtered_labels)
                show_labeled_image(frame, filtered_boxes, filtered_labels, filtered_scores, img_id, output_path)
                print("time to prediction:", end - start)

            end = time()
            print("time to prediction:", end - start)

        else:
            break

        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


def map_barcode_to_labels(barcode_labels, predictions):
    id_order = {id: i for i, id in enumerate(barcode_labels)}
    new_label = 0
    labels = []
    for label in predictions:
        id = label.split('_')[0]
        if id not in id_order:
            id_order[id] = new_label
            new_label += 1
        labels.append(str(id_order[id]))

    return labels
