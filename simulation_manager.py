from multiprocessing import Pool
import os
from re import search as re_search
import Inference
import pandas as pd


def wright_fisher():
    if __name__ == '__main__':
        pool = Pool(13)  # 13 for ~95% usage
        inputs = [x for x in range(1, 101)]
        outputs = pool.map(WF.main, inputs)
        return outputs


def selection_inference():
    if __name__ == "__main__":
        folder = "Data"
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        pool = Pool(13)
        outputs = pool.map(Inference.main, files)
        data = outputs
        df = pd.DataFrame(data)
        df.to_csv("Selection Inference Results.csv")


def mutation_inference(folder):
    if __name__ == "__main__":
        s_no = folder.split("s#")[-1]
        s = re_search("_s_(.*)_s#", folder).group(1)
        print("Working on {0} S{1}".format(s_no, s))
        mut_data = [dict({"File Name": os.path.join(folder, f), "# of Selection": int(s_no), "Selection": float(s)})
                    for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        pool = Pool(13)
        outputs = pool.map(Inference.main, mut_data)
        data = outputs
        df = pd.DataFrame(data)
        df.to_csv("{0}.csv".format(folder), index=False)


samples = [(400, 1, 50), (250, 1, 50), (400, 1, 20), (250, 1, 20),
           (400, 2.5, 50), (250, 2.5, 50), (400, 2.5, 20), (250, 2.5, 20)]

# params = samples[0]
# 0

# folder = "Problems/G400_L5_M_0.001_s_0.01_s#2"
# result = folder.split("/")[-1]
# mut_data = [dict({"File Name": os.path.join(folder, f), "# of Selection": 2, "Selection": 0.01})
#             for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


main_dir = "More Data"
folders = [os.path.join(main_dir, f) for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
for idx, directory in enumerate(folders):
    mutation_inference(folder=directory)

# mutation_inference(params=mut_data, save_file=result)