from colordescriptor import ColorDescriptor
from searcher_new import Searcher
import cv2

# Hard-coded values (developer can change these directly)
index_path = "/path/to/index/file"  # Path to the precomputed index
query_image_path = "/path/to/query/image.png"  # Path to the query image
result_path = "/path/to/result/images"  # Path to store/display results

# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# load the query image and describe it
query = cv2.imread(query_image_path)
features = cd.describe(query)

# perform the search
searcher = Searcher(index_path)
results = searcher.search(features)

# display the query
cv2.imshow("Query", query)

# loop over the results
for (score, resultID) in results:
    # load the result image and display it
    result_image_path = f"{result_path}/{resultID}"
    result = cv2.imread(result_image_path)
    cv2.imshow("Result", result)
    cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
