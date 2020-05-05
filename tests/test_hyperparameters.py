from colorizexl import ColorizeXL
import cv2
import time
import pandas as pd

filename_bw = "./samples/girl_grayscale.png"
filename_annotated = "./samples/girl_annotated5.png"
recolorize = False
output_name = ""
ground_truth_filename = "./samples/girl.jpg"
output_folder = "./output_20200502/"

neighbours = [1, 5, 10, 15, 20]
overlaps = [1, 2, 4, 8]
patch_sizes = [30, 60, 120, 240]

ground_truth = cv2.imread(ground_truth_filename) / 255.0

df_all = pd.DataFrame()

for neighbour in neighbours:
    for overlap in overlaps:
        for patch_size in patch_sizes:
            if neighbour * 2 + 1 > patch_size:  # TODO: raise exception in code
                continue

            print(neighbour, overlap, patch_size)
            colorizer = ColorizeXL(filename_bw, filename_annotated, recolorize)
            start = time.time()
            output = colorizer.colorize(
                step_size=patch_size, overlap=overlap, n=neighbour
            )
            end = time.time()
            elapsed = end - start
            output_name = (
                output_folder + "out_"
                + str(neighbour)
                + "_"
                + str(overlap)
                + "_"
                + str(patch_size)
                + ".png"
            )
            cv2.imwrite(
                output_name,
                cv2.cvtColor(output.astype("float32"), cv2.COLOR_BGR2RGB) * 255.0,
            )

            ssim, mse, pnsr = colorizer.compare_images_metrics(
                img=output, gt=ground_truth
            )
            df = pd.DataFrame(
                [
                    [
                        neighbour,
                        overlap,
                        patch_size,
                        output_name,
                        elapsed,
                        ssim,
                        mse,
                        pnsr,
                    ]
                ]
            )
            df_all = df_all.append(df)

df_all.columns = [
    "neighbour",
    "overlap",
    "patch_size",
    "output_name",
    "elapsed",
    "ssim",
    "mse",
    "pnsr",
]
df_all.to_csv(output_folder + "results_all_20200503.csv", index=False)
