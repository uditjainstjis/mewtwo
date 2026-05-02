import os

BASE_DIR = "/home/learner/Desktop/mewtwo"
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive/research")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

GROUPS = {
    "MASTER_KNOWLEDGE_BASE.md": [
        "Research_Context_Knowledge_Base.md",
        "Research_Context_Knowledge_Base1.md",
        "Token_Level_Routing_Research_KB.md"
    ],
    "MASTER_RESEARCH_CHRONICLES.md": [
        "MEWTWO_COMPLETE_RESEARCH_CHRONICLE.md",
        "Nemotron_30B_Research_Chronicle.md",
        "Post_KB_Research_Chronicle.md",
        "THE_MEWTWO_CHRONICLES.md"
    ],
    "MASTER_EXPERIMENT_REPORTS.md": [
        "FINAL_CONCLUSION_NOTE_2026_04_09.md",
        "FINAL_EXPERIMENT_REPORT_2026_04.md",
        "lori_moe_validation_report.md",
        "newest_experiment.txt",
        "research_results.md",
        "research_summary.md",
        "why we not good😭.md"
    ],
    "MASTER_TASKS_AND_PLANS.md": [
        "NEMOTRON_PLAN.md",
        "NEMOTRON_TASKLIST.md",
        "PIPELINE_TASKS.md",
        "implementation_plan.md"
    ]
}

def consolidate():
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    for master_filename, source_files in GROUPS.items():
        master_filepath = os.path.join(DOCS_DIR, master_filename)
        
        toc = [f"# Table of Contents for {master_filename}\n"]
        content_blocks = []
        
        for sf in source_files:
            sf_path = os.path.join(ARCHIVE_DIR, sf)
            if not os.path.exists(sf_path):
                print(f"Warning: {sf} not found in {ARCHIVE_DIR}")
                continue
                
            with open(sf_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            header_title = sf.replace("_", " ").replace(".md", "").replace(".txt", "")
            anchor = header_title.lower().replace(" ", "-")
            
            toc.append(f"- [{header_title}](#source-{anchor})")
            
            block = f"## Source: {header_title}\n\n{content}\n"
            content_blocks.append(block)
            
        with open(master_filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(toc))
            f.write("\n\n---\n\n")
            f.write("\n\n---\n\n".join(content_blocks))
            
        print(f"Created {master_filepath} from {len(content_blocks)} files.")

if __name__ == "__main__":
    consolidate()
