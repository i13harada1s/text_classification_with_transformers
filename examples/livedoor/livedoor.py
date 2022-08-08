import os
import csv

from sklearn.model_selection import train_test_split


def format_csv(tgt_dir: str, out_path: str):
    if not os.path.exists(tgt_dir):
        raise ValueError(f"not found the directory: {tgt_dir}")
    
    with open(out_path, "w", encoding="utf-8-sig") as o:
        w = csv.writer(o)
        w.writerow(["label", "text"]) # make headder
        for name1 in os.listdir(tgt_dir):
            # category
            ctg_dir = os.path.join(tgt_dir, name1)
            if os.path.isdir(ctg_dir):
                for name2 in os.listdir(ctg_dir):
                    # document
                    txt_file = os.path.join(ctg_dir, name2)
                    with open(txt_file, "r", encoding="utf-8") as r:
                        # 0: url, 1: timestamp
                        txt = r.readlines()[2:]
                        txt = [t.strip() for t in txt]
                        txt = list(filter(lambda line: line != "", txt))
                        w.writerow([name1, "".join(txt)])


def read_csv(in_path: str):
    if not os.path.exists(in_path):
        raise ValueError(f"not found the file: {in_path}")
    with open(in_path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        dict_list = [d for d in r]
    return dict_list

def output_csv(dict_list, out_path: str):
    headder = dict_list[0].keys()
    with open(out_path, "w", encoding="utf-8-sig") as o:
        w = csv.DictWriter(o, fieldnames=headder)
        w.writeheader()
        w.writerows(dict_list)

if __name__ == "__main__":
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    format_csv("tmp/text", "tmp/lbd.csv")
    data = read_csv("tmp/lbd.csv")
    train, holdout = train_test_split(data, test_size=(1-train_ratio), random_state=42)
    valid, test = train_test_split(holdout, test_size=test_ratio/(test_ratio+valid_ratio), random_state=42)
    output_csv(train, "tmp/train.csv")
    output_csv(valid, "tmp/valid.csv")
    output_csv(test, "tmp/test.csv")