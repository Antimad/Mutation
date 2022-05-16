from multiprocessing import Pool
import os
import WF
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


def mutation_inference():
    if __name__ == "__main__":
        folder = "Data"
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        pool = Pool(13)
        outputs = pool.map(Inference.main, files)
        data = outputs
        df = pd.DataFrame(data)
        df.to_csv("Mutation Inference Results.csv")
