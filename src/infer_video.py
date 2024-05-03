import argparse
import tempfile

import cv2
import numpy as np
import torch

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description="Infer a detector")
    parser.add_argument("config", help="config file path")
    parser.add_argument("imgname", help="image file name")

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp="out.jpg"):
    font_scale = 0.5
    bbox_color = "green"
    text_color = "green"
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = (
            class_names[label] if class_names is not None else f"cls {label}"
        )
        if len(bbox) > 4:
            label_text += f"|{bbox[-1]:.02f}"
        cv2.putText(
            img,
            label_text,
            (bbox_int[0], bbox_int[1] - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            text_color,
        )
    imwrite(img, outfp)


def save_results(result, img_shape, frame_num, time_elapsed, output_file="output.csv"):
    # convert bboxes to standard [x1, y1, x2, y2] format
    # with each as a float from 0 to 1
    bboxes = np.vstack(result)
    bboxes = bboxes[:, :4]
    bboxes /= np.array([img_shape[1], img_shape[0], img_shape[1], img_shape[0]])

    # bboxes of shape (n, 5), where n is the number of detections
    # we only care for the first 4 values
    # output to csv of format: <framenumber>,<time_elapsed>,[x1,y1,x2,y2],...
    output = []
    for bbox in bboxes:
        output.append(f'"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"')

    str_output = f'"{frame_num}","{time_elapsed}",[' + ",".join(output) + "]\n"
    with open(output_file, "a") as f:
        f.write(str_output)


def main():

    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg")
    video_source = "/home/ani/dev/projects/face-blur-rt/data/demos/multi.mp4"
    cap = cv2.VideoCapture(video_source)
    frame_num = 0

    args = parse_args()
    cfg = Config.fromfile(args.config)
    # imgname = args.imgname

    imgname = temp_file.name
    class_names = cfg.class_names

    engine, data_pipeline, device = prepare(cfg)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        with open(temp_file.name, "wb") as f:
            f.write(cv2.imencode(".jpg", frame)[1].tobytes())

        data = dict(img_info=dict(filename=imgname), img_prefix=None)

        data = data_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if device != "cpu":
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            # just get the actual data from DataContainer
            data["img_metas"] = data["img_metas"][0].data
            data["img"] = data["img"][0].data

        time_start = torch.cuda.Event(enable_timing=True)
        time_start.record()

        result = engine.infer(data["img"], data["img_metas"])[0]

        time_end = torch.cuda.Event(enable_timing=True)
        time_end.record()
        torch.cuda.synchronize()
        time_elapsed = time_start.elapsed_time(time_end)
        print(time_elapsed)

        save_results(
            result,
            data["img_metas"][0][0]["ori_shape"],
            frame_num,
            time_elapsed,
            output_file="output.csv",
        )

        # plot_result(result, imgname, class_names)

        frame_num += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
