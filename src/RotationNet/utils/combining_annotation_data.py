# %%
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RotationNet.utils.image_utils import rotate_image


# %%
def compute_angle(x1, y1, x2, y2):
    """Compute angle between two points."""
    return -np.degrees(np.arctan2(y2 - y1, x1 - x2))


# %%
def get_full_image_path(name):
    """Get full image path."""
    return os.path.join("/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/small_images_flat/", name)


# %%
def convert_full_image_path_to_scad_path(path):
    """Convert full image path to SCAD path."""
    prefix = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/small_images/"
    path = path[len(prefix) :].split(".")[0]
    path = path + "." + images_on_scad_extension_map[path]
    return "Z:\\" + path.replace("/", "\\")


# %%
def parse_xml_and_calculate_angles(file_path):
    """Parse XML and calculate angles."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    lines_data = []
    for image in root.iter("image"):
        image_name = image.get("name")
        for polyline in image.findall("polyline"):
            points_data = polyline.get("points").split(";")
            if len(points_data) == 2:
                x1, y1 = map(float, points_data[0].split(","))
                x2, y2 = map(float, points_data[1].split(","))
                if x1 > x2:
                    angle = compute_angle(x1, y1, x2, y2)
                    destination = get_full_image_path(image_name)
                    source = images_map_dict[destination]
                    scad_file_path = convert_full_image_path_to_scad_path(source)
                    annotations = {
                        "scad_path": scad_file_path,
                        "della_source": source,
                        "flat_file_location": destination,
                        "angle": angle,
                    }
                    lines_data.append(annotations)

    return lines_data


# %%
images_on_scad_path = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/indexes/all_images_on_scad"
ground_truth_data_path = (
    "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/skew_information/ground_truth_image_data.csv"
)
images_map = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/small_images_flat/file_mapping"


# %%
with open(images_on_scad_path, "r") as file:
    images_on_scad = file.read().split("\n")

images_on_scad_extension_map = {
    image.split(".")[0].replace("\\", "/"): image.split(".")[-1] for image in images_on_scad
}

ground_truth_df = pd.read_csv(ground_truth_data_path)
ground_truth_df["normalized_path"] = (
    ground_truth_df["file_path"].str.replace("\\", "/").str[3:].str.replace(".tif", ".jpg")
)

images_map_df = pd.read_csv(images_map, header=None)
images_map_df.columns = ["mapping"]
images_map_df["mapping"] = images_map_df["mapping"].str.replace("\\", "/")
images_map_df[["source", "destination"]] = images_map_df["mapping"].str.split(" -> ", expand=True)
images_map_df = images_map_df.drop(columns=["mapping"])

prefix = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/small_images/"
images_map_df["source_no_prefix"] = images_map_df["source"].str[len(prefix) :]

images_map_dict = images_map_df.set_index("destination")["source"].to_dict()


# %%
merged_df = ground_truth_df.merge(images_map_df, left_on="normalized_path", right_on="source_no_prefix")
keep = ["file_path", "source", "destination", "angle"]
rename = {"file_path": "scad_path", "source": "della_source", "destination": "flat_file_location"}
merged_df = merged_df[keep].rename(columns=rename)


# %%
files = [
    "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/skew_information/1",
    "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/skew_information/2",
    "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/skew_information/3",
]
xml_annotations = [annotation for file in files for annotation in parse_xml_and_calculate_angles(file)]
xml_annotations_df = pd.DataFrame.from_dict(xml_annotations)

merged_df = pd.concat([merged_df, xml_annotations_df], ignore_index=True)
merged_df.to_csv(
    "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/skew_information/small_images_flat_image_annotations.csv",
    index=False,
)

# %%
merged_df["shelfmark"] = merged_df["scad_path"].str[3:].str.split(".").str[0]
merged_df["front"] = merged_df["shelfmark"].str.split("_").str[-1] == "r"
merged_df["shelfmark"] = merged_df["shelfmark"].str.split("_").str[:-1].str.join("_")


def move_images_to_processed_folder(df):
    """Move images to processed folder."""
    destination_folder = "/scratch/gpfs/RUSTOW/deskewing_datasets/images/jts_images/small_images_flat_processed"
    count = sum(
        1
        for image_source in df["flat_file_location"]
        if os.path.exists(image_source)
        and shutil.move(image_source, os.path.join(destination_folder, os.path.basename(image_source)))
    )
    print(f"moved {count} images")


# %%
move_images_to_processed_folder(merged_df)


# %%
def show_image(i, df):
    image_path = df["flat_file_location"][i].replace("small_images_flat", "small_images_flat_processed")
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = rotate_image(image, np.radians(df["angle"][i]))
    plt.imshow(image)
    plt.show()


# %%
non_zero_angles = merged_df[merged_df["angle"] != 0].reset_index(drop=True)
i = np.random.randint(0, len(non_zero_angles))
show_image(i, non_zero_angles)


# %%
# show group i using get_group
def show_images_from_shelfmark(df, idx):
    grouped = df.groupby("shelfmark")
    grouped = grouped.filter(lambda x: len(x) == 2).groupby("shelfmark")
    group_i_name = list(grouped.groups.keys())[idx]

    group = grouped.get_group(group_i_name)
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))

    for i in range(2):
        image_path = group.iloc[i]["flat_file_location"].replace("small_images_flat", "small_images_flat_processed")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        angle = round(group.iloc[i]["angle"], 2)
        image = rotate_image(image, np.radians(angle))
        axs[i].imshow(image)
        axs[i].set_title(f"Image {i+1} from {group_i_name}")
        axs[i].text(
            0.5,
            0.5,
            f"Angle: {angle}",
            fontsize=20,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[i].transAxes,
        )

    plt.show()


# %%
group_no = np.random.randint(0, 1000)
show_images_from_shelfmark(non_zero_angles, group_no)

# %%
