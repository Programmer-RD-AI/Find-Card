from tqdm import tqdm
import urllib.request
import pandas as pd

new_data = {"Path": [], "XMin": [], "YMin": [], "XMax": [], "YMax": [], "ImageID": []}
idx = 0
data = pd.read_csv("./V2/data/Cleaned-Data.csv")
for img_url, xmin, ymin, xmax, ymax, ourl in tqdm(
    zip(
        data["ImageID"],
        data["XMin"],
        data["YMin"],
        data["XMax"],
        data["YMax"],
        data["OriginalURL"],
    )
):
    try:
        idx += 1
        # print(img_url, xmin, ymin, xmax, ymax, ourl)
        urllib.request.urlretrieve(ourl, f"./V2/Img/{idx}.png")
        new_data["Path"].append(f"{idx}.png")
        new_data["XMin"].append(xmin)
        new_data["YMin"].append(ymin)
        new_data["XMax"].append(xmax)
        new_data["YMax"].append(ymax)
        new_data['ImageID'].append(img_url)
    except Exception as e:
        pass
# print(new_data)
data = pd.DataFrame(new_data)
data.to_csv("./V2/Data.csv", index=False)
