import collections
import glob
import json
import os
import re
import sys
import zipfile


class SuperGeluResult:
    def __init__(self):
        self.boolq_set = []
        self.cb_set = []
        self.copa_set = []
        self.multirc_set = collections.defaultdict(dict)
        self.record_set = []
        self.rte_set = []
        self.wic_set = []
        self.wsc_set = []
        self.indexs_set = {}

        self.ax_g_set = [{"idx": i, "label": "entailment"} for i in range(356)]
        self.ax_b_set = [{"idx": i, "label": "entailment"} for i in range(1104)]

    def synthesis(self, path, output_dir):
        search_re = re.compile(r"<l>(.*)</l>")
        search_re_idx = re.compile(r"<idx>(.*)</idx>")

        # load in all the json data
        f = open(path, "r")
        line = f.readline()

        while len(line) > 0:
            tokens = json.loads(line)
            info = tokens["ground_truth"]

            origin = search_re.search(info).group(1)
            indexes = search_re_idx.search(info).group(1)
            result = tokens["answer"]

            if origin != "multirc":
                fn = getattr(self, "common", None)
            else:
                fn = getattr(self, "multirc", None)
            if fn:
                fn(origin, indexes, result)

            line = f.readline()

        self.terminator(output_dir)
        print("Success.")

    def common(self, origin, indexes, result):
        idx = json.loads(indexes)["idx"]
        if origin not in self.indexs_set:
            self.indexs_set[origin] = set()
        if origin == "record":
            idx = idx["query"]

        if origin == "copa":
            result = int(result) - 1

        if origin != "record" and isinstance(result, str):
            result = result.strip()
            if origin != "wsc":
                result = result.lower()
        if origin == "rte":
            result = "not_entailment" if result == "no" else "entailment"

        res = {"idx": idx, "label": result}
        selected_set = getattr(
            self, origin.lower().replace("-", "_") + "_set", None
        )
        if idx not in self.indexs_set[origin]:
            selected_set.append(res)
            self.indexs_set[origin].add(idx)

    def multirc(self, origin, indexes, result):
        idx = json.loads(indexes)["idx"]
        passage_id = idx["paragraph"]

        cur_dict = self.multirc_set[passage_id]
        cur_dict["idx"] = passage_id
        if "passage" not in cur_dict:
            cur_dict["passage"] = {"questions": []}
            cur_dict["passage"]["questions"].append(
                {"idx": idx["question"], "answers": []}
            )

            cur_dict["passage"]["questions"][0]["answers"].append(
                {
                    "idx": idx["answer"],
                    "label": 1 if result.strip().lower() == "true" else 0,
                }
            )
        else:
            flag = False
            for q in cur_dict["passage"]["questions"]:
                if q["idx"] == idx["question"]:
                    flag = True
                    q["answers"].append(
                        {
                            "idx": idx["answer"],
                            "label": 1
                            if result.strip().lower() == "true"
                            else 0,
                        }
                    )
                    break
            if not flag:
                cur_dict["passage"]["questions"].append(
                    {"idx": idx["question"], "answers": []}
                )
                cur_dict["passage"]["questions"][-1]["answers"].append(
                    {
                        "idx": idx["answer"],
                        "label": 1 if result.strip().lower() == "true" else 0,
                    }
                )

    def zipdir(self, output_dir):
        paths = output_dir.split(os.path.sep)

        filename = "_".join(paths[-4:]) + ".zip"
        filename = filename.replace("-", "_")
        output_zip_dir = os.path.join(os.path.sep, *paths[:-1], filename)
        zip_dir = zipfile.ZipFile(
            output_zip_dir,
            "w",
            zipfile.ZIP_DEFLATED,
        )
        for path, dirnames, filenames in os.walk(output_dir):
            fpath = path.replace(output_dir, "")
            for filename in filenames:
                zip_dir.write(os.path.join(path, filename), fpath + filename)
        zip_dir.close()

    def terminator(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        new_multirc_set = []
        for k, v in self.multirc_set.items():
            new_multirc_set.append(v)
        self.multirc_set = new_multirc_set

        # output all required data
        file_names = [
            "BoolQ",
            "CB",
            "COPA",
            "MultiRC",
            "ReCoRD",
            "RTE",
            "WiC",
            "WSC",
            "AX-g",
            "AX-b",
        ]
        for file in file_names:
            selected_set = getattr(
                self, file.lower().replace("-", "_") + "_set", None
            )
            with open(
                os.path.join(output_dir, file + ".jsonl"), "w", encoding="utf-8"
            ) as fout:
                for item in selected_set:
                    json_str = json.dumps(item)
                    fout.write(json_str + "\n")
            fout.close()

        self.zipdir(output_dir)


if __name__ == "__main__":
    import glob

    path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/super_glue_all"
    filenames = glob.glob(f"{path}/**/*result_predict.jsonl", recursive=True)

    for filename in filenames:
        s = SuperGeluResult()
        s.synthesis(
            filename,
            os.path.join(os.path.dirname(filename), "super_glue"),
        )

    x = 1

    # s = SuperGeluResult()

    # s.synthesis(
    #     path,
    #     os.path.join(os.path.dirname(path), "super_glue"),
    # )
