import argparse
import time
from sys import platform
import os

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        data_cfg,
        weights,
        images='data/samples',  # input folder
        output='output',  # output folder
        fourcc='mp4v',
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=True,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    torch.backends.cudnn.benchmark = False  # set False for reproducible results
    if os.path.exists(output) == False:
        os.makedirs(output)  # make new output folder

    # Initialize model
    if ONNX_EXPORT:
        s = (320, 192)  # (320, 192) or (416, 256) or (608, 352) onnx model image size (height, width)
        model = Darknet(cfg, s)
    else:
        model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    if ONNX_EXPORT:
        img = torch.zeros((1, 3, s[0], s[1]))
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=True)
        return

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    num_detected = 0

    label_output = os.path.join(output, 'pred')
    image_output = os.path.join(output, 'images')
    if os.path.exists(label_output) == False:
        os.mkdir(label_output)
    if os.path.exists(image_output) == False:
        os.mkdir(image_output)

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()

        txt_save_path = str(Path(label_output) / Path(path).name)
        im_save_path = str(Path(image_output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if det is not None and len(det) > 0:
            # add 1 to num_detected
            num_detected += 1

            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to screen
            print('%gx%g ' % img.shape[2:], end='')  # print image size
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in det:
                if save_txt:  # Write to file
                    with open(txt_save_path.replace('png', 'txt'), 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save image with detections
            if dataloader.mode == 'images':
                cv2.imwrite(im_save_path, im0)

    print('\nnumber of detected dams: ', num_detected, '\n')
    print('Results saved to %s' % os.getcwd() + os.sep + output)


    #if save_images:
        
     #   if platform == 'darwin':  # macos
      #      os.system('open ' + output + ' ' + im_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
    parser.add_argument('--output', type=str, default='output', help='specifies the output path for images and videos')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(opt.cfg,
               opt.data_cfg,
               opt.weights,
               images=opt.images,
               img_size=opt.img_size,
               conf_thres=opt.conf_thres,
               nms_thres=opt.nms_thres,
               fourcc=opt.fourcc,
               output=opt.output)
