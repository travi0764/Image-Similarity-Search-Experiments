# import the necessary packages
from colordescriptor import ColorDescriptor
import glob
import cv2
from tqdm.auto import tqdm

# Manually specify the paths here
dataset_path = "data/train"  # Path to the directory that contains the images
index_path = "data/index.csv"  # Path to where the computed index will be stored

# initialize the color descriptor
cd = ColorDescriptor((32, 32, 3))

# open the output index file for writing
output = open(index_path, "w")

# use glob to grab the image paths and loop over them
for imagePath in glob.glob(dataset_path + "/*.png"):
    # extract the image ID (i.e. the unique filename) from the image
    # path and load the image itself
    imageID = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    
    # describe the image
    features = cd.describe(image)
    
    # write the features to file
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the index file
output.close()

print(f"Indexing complete. The features have been saved to {index_path}.")
