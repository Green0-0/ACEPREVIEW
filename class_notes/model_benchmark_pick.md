Model: cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit, lowest thinking level (no thinking), standard samplers, served with vLLM across a5000s on slurm (should fit on 1, but just barely, maybe a few thousand t/s) with PP. NO use of slurm lab (bhuwan, nlplab) GPUs.
Comparisons/Harness: Standard agent2 agent (all tools, agent2 standard pipeline) vs NEAT learned agent2 agent (agent2 custom graph pipeline)
Benchmark: SWE Bench Verified
Training Data: internlm/SWE-Fixer-Train-110K (Use mini batches to accelerate training/reduce overfitting, so the data changes for each population, this is okay). If this dataset does not work something else that is similar is also okay.
The goal is described in project_question.md