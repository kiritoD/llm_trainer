import json

from src.utils.logging import get_logger
from src.utils.metric import boolq_acc, xsum_metrics

logger = get_logger(__name__)

neg_label = "非负向"
# print(sys.argv[1])
# path = sys.argv[1]
path = [
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/adalora/adalora_xsum_cosine_original_r16_8_42/ADALORA/step_314/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/adalora/adalora_xsum_cosine_original_r16_8_42/ADALORA/step_392/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/adalora/adalora_xsum_cosine_original_r16_8_42/ADALORA/step_468/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/adalora/adalora_xsum_cosine_original_r8_4_42/ADALORA/step_314/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/adalora/adalora_xsum_cosine_original_r8_4_42/ADALORA/step_392/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/adalora/adalora_xsum_cosine_original_r8_4_42/ADALORA/step_468/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r2/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r2/LORA/step_160/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r2/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r2/LORA/step_200/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r4/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r4/LORA/step_160/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r4/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r4/LORA/step_200/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r8/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r8/LORA/step_80/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r8/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r8/LORA/step_120/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r8/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r8/LORA/step_160/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r8/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r8/LORA/step_200/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r16/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r16/LORA/step_80/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r16/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r16/LORA/step_120/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r16/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r16/LORA/step_160/result_predict.jsonl",
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/rank_experiments/llama2/xsum/r16/peft_lora_lr1e-3_train_xsum_cosine_d_parts_r16/LORA/step_200/result_predict.jsonl",
]
path_length = len(path)
with open(
    "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/peft_llm/model_output/inference/results_temp.jsonl",
    "w",
) as f:
    metrics = []
    for index, path_ in enumerate(path):
        logger.info(
            f"[{index + 1}/{path_length}]: start to caculate the result about the {path_} \n"
        )
        metric = xsum_metrics(path_)
        metric.update({"filename": path_})
        logger.info(f"{metric}\n")
        metrics.append(metric)
    metrics = sorted(metrics, key=lambda x: x["rouge1"], reverse=True)
    for metric_ in metrics:
        f.write(f"{json.dumps(metric_)}\n")
    logger.info("over~")


# path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/fashion_bloom/model_outputs/BELLE_7B_2M_peft/remote-lr8e-4-bs32-ep5-peft_lora_linear_loss_mask_causual_loss_False_lambda_0.1/lora/global_step1/results_remote.jsonl"
# path = "/mnt/bn/ecom-ccr-dev/mlx/users/dongjunwei/EasyGuard/examples/fashion_bloom/model_outputs/BELLE_7B_2M_peft/remote-lr2e-4-bs32-ep5-peft_lora_linear_loss_mask_causual_loss_ccr_v3_False_lambda_0.1/lora/global_step5/results_live_remote.jsonl"
# a = []
# with open(path, "r") as f:
#     for line in f:
#         a.append(json.loads(line.strip()))
# y_trues, y_preds, label_cnts = load_result_from_list(a)

# evaluate_model(
#     y_trues=y_trues,
#     y_preds=y_preds,
#     label_cnts=label_cnts,
#     cut=10,
#     ignore_labels=[neg_label],
# )

# threhold = 0.8
# # recall_pre(threhold)
# # print(
# #     evaluate4jsonl(
# #         path,
# #         "/mnt/bn/ecom-ccr-dev/mlx/users/lida/unified/ccr_all_train_v3_with_source_0522_4_train.parquet",
# #     )
# # )
# print(boolq_acc(path))
