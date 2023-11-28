"""
[Warning] DO NOT use this script for REPLUG+LLaMA result calculation.

This script uses macro-average as it copied code-davinci-002 results from the FLAN-PaLM paper. 
"""
import collections
import fire
import numpy as np
import pandas as pd


MMLU_TASKS_DOMAINS = {
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "Other",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Science",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "Humanities",
    "global_facts": "Other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Science",
    "high_school_government_and_politics": "Social Science",
    "high_school_macroeconomics": "Social Science",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Science",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Science",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Other",
    "human_sexuality": "Social Science",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "machine_learning": "STEM",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "Other",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "Other",
    "professional_law": "Humanities",
    "professional_medicine": "Other",
    "professional_psychology": "Social Science",
    "public_relations": "Social Science",
    "security_studies": "Social Science",
    "sociology": "Social Science",
    "us_foreign_policy": "Social Science",
    "virology": "Other",
    "world_religions": "Humanities",
}


def main(result_csv: str):
    df = pd.read_csv(result_csv)

    cat_results = []
    for i, row in df.iterrows():
        if isinstance(row["subject"], str) and "davinci" in row["subject"]:
            cat_ems = collections.defaultdict(list)
            cat_ems["model"] = row["subject"]
            for subject in row.keys():
                if subject not in ["subject", "overall"]:
                    _subject = subject.replace("_results", "")
                    cat = MMLU_TASKS_DOMAINS[_subject]
                    cat_ems[cat].append(float(row[subject]))
                                
            for key in cat_ems:
                if key != "model":
                    cat_ems[key] = np.mean(cat_ems[key])
            cat_results.append(cat_ems)

    cat_results = pd.DataFrame(cat_results)
    out_csv = result_csv.split(".csv")[0] + ".cat.csv"
    cat_results.to_csv(out_csv, float_format="%.1f", index=False)
    print(f"Categorized results saved to {out_csv}")


if __name__ == "__main__":
    fire.Fire(main)