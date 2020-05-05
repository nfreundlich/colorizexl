from colorizexl import ColorizeXL
import cv2
import time
import pandas as pd
import numpy as np

filename_bw = "./samples/girl_large_grayscale.png"
filename_annotated = "./samples/girl_large_grayscale_annotated4.png"
recolorize = False
output_name = ""
filename_ground_truth = "./samples/girl_large.jpg"

neighbours = [1, 5, 10, 15, 20]
overlaps = [1, 2, 4, 8]
patch_sizes = [30, 60, 120, 240]

# read large images
ground_truth = cv2.imread(filename_ground_truth) / 255.0
im_grayscale = cv2.imread(filename_bw)
im_annotation = cv2.imread(filename_annotated)

df_all = pd.DataFrame()

neighbour = 10
overlap = 2
patch_size = 60

input_size = np.arange(128, 1280, step=128)

for factor in np.arange(0.1, 1.1, step=0.1):
    # scaled filenames
    input_grayscale_name = "./output_timing_no_patch/girl_grayscale_" + str(factor) + ".png"
    input_annotation_name = "./output_timing_no_patch/girl_annotation_" + str(factor) + ".png"
    ground_truth_small_name = "./output_timing_no_patch/girl_gt_" + str(factor) + ".png"

    # rescale grayscale image

    im_grayscale_small = cv2.resize(
        im_grayscale, dsize=(0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
    )
    cv2.imwrite(input_grayscale_name, im_grayscale_small)

    # rescale annotated image
    im_annotation_small = cv2.resize(
        im_annotation, dsize=(0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
    )
    cv2.imwrite(input_annotation_name, im_annotation_small)

    # rescale ground truth
    im_gt_small = cv2.resize(
        ground_truth, dsize=(0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
    )

    # imsize
    h, w, c = ground_truth.shape

    # call colorizer
    start = time.time()
    colorizer = ColorizeXL(
        input_grayscale_name, input_annotation_name, recolorize=False
    )
    output = colorizer.colorize_no_patch()
    end = time.time()
    elapsed = end - start

    # save output image
    output_name = "./output_timing_no_patch/out_" + str(factor) + ".png"
    cv2.imwrite(
        output_name, cv2.cvtColor(output.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0,
    )

    df = pd.DataFrame([[factor, output_name, elapsed]])
    df_all = df_all.append(df)


    ##df_all.columns = [
    #    "factor",
    #    "output_name",
    #    "elapsed"
    #]
    df_all.to_csv("./output_timing_no_patch/results_timing_no_patch.csv", index=False)
