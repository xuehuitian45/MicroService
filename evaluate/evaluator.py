from evaluate.config import DataConfig, EvaluateConfig
from evaluate.model import EvaluateResult
from evaluate.semantic.comparative_evaluate import main as semantic_main
from evaluate.structure.structured_evaluate import main as structured_main
import os

def calculate_evaluation_metrics():
    data_config = DataConfig()
    evaluation_config = EvaluateConfig()
    structured_result = structured_main(data_config)
    semantic_result = semantic_main(data_config.result_path, evaluation_config.repeat_times)
    report = {}
    count = 0
    for folder in os.listdir(data_config.result_path):
        folder_path = os.path.join(data_config.result_path, folder)
        if os.path.isdir(folder_path):
            report[folder] = EvaluateResult(
                SC=semantic_result["SC"][count],
                SCP=semantic_result["SCP"][count],
                SBC=semantic_result["SBC"][count],
                judge_result=semantic_result["judge_result"][count],
                ISDD=structured_result[count]["ISDD"],
                SDE=structured_result[count]["SDE"],
                SSB=structured_result[count]["SSB"],
                R_nano=structured_result[count]["R_nano"],
                R_mega=structured_result[count]["R_mega"],
                SII=structured_result[count]["SII"],
                ICP=structured_result[count]["ICP"],
                Modularity=structured_result[count]["Modularity"],
                SCDR=structured_result[count]["SCDR"],
            )
            count+=1

    return report