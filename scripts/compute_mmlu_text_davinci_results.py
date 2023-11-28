"""
Collect subject-wise MMLU results from the text-davinci-003 REPLUG experiments.
"""
import getpass
import glob
import json
import numpy as np
import os
import pandas as pd


result_swj_dir = f"/checkpoint/{getpass.getuser()}/retrofit/results_swj/"
open_book_results_dir = os.path.join(result_swj_dir, "mmlu/")
closed_book_results_dir = os.path.join(result_swj_dir, "mmlu.closed_book/")


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


def main():
    open_metrics = {"__run_name": "text-davinci-003 open-book"}
    closed_metrics = {"__run_name": "text-davinci-003 closed-book"}
    count_metrics = {"__run_name": "count"}

    open_ems = []
    closed_ems = []
    num_subjects = 0

    for open_in_json in glob.glob(os.path.join(open_book_results_dir, "*.json")):
        if "overall_results.json" in open_in_json:
            continue
        subject = os.path.basename(open_in_json).replace(".json", "")
        cat = MMLU_TASKS_DOMAINS[subject]
        
        with open(open_in_json) as f:
            open_results = json.load(f)
            open_subj_ems = [pred["em"] for pred in open_results["predictions"]]
        open_subj_em = np.mean(open_subj_ems) * 100
        count_metrics[subject] = int(len(open_subj_ems))
        open_metrics[subject] = open_subj_em
        open_ems.extend(open_subj_ems)

        closed_in_json = os.path.join(closed_book_results_dir, os.path.basename(open_in_json))
        with open(closed_in_json) as f:
            closed_results = json.load(f)
            closed_subj_ems = [pred["em"] for pred in closed_results["predictions"]]
        assert(len(open_subj_ems) == len(closed_subj_ems))
        closed_subj_em = np.mean(closed_subj_ems) * 100
        closed_metrics[subject] = closed_subj_em
        closed_ems.extend(closed_subj_ems)

        num_subjects += 1
        print(f"{num_subjects}. {open_in_json}")
        print(f"Open Subj EM: {open_subj_em:.2f}")
        print(f"Closed Subj EM: {closed_subj_em:.2f}")
        print()

    open_em = np.mean(open_ems) * 100
    closed_em = np.mean(closed_ems) * 100
    open_metrics["overall"] = open_em
    closed_metrics["overall"] = closed_em

    results_df = pd.DataFrame([count_metrics, open_metrics, closed_metrics])
    out_csv = os.path.join(result_swj_dir, "text-davinci-003.tsv")
    results_df.to_csv(out_csv, float_format="%.1f")
    print(f"Results saved to {out_csv}")
    
    print(f"Open EM: {open_em:.2f}, {len(open_ems)} predictions")
    print(f"Closed EM: {closed_em:.2f}, {len(closed_ems)} predictions")


if __name__ == "__main__":
    main()