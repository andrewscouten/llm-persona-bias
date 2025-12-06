## Introduction

This project is the final assignment for the course and replaces the final exam. The goal is for you to work on a machine learning problem in depth. You will formulate a question, design experiments, implement models, and report your findings. This process will help you apply the concepts from the course to a real problem. You will work in groups of two. 

## The First Step: Project Proposal

The first deliverable is a project proposal. You have one week to form your group, choose a project, and write a 1-2 page proposal.

The purpose of the proposal is to ensure your project's scope is reasonable. It should not be too hard or too easy. The instructor and TA will review your proposal and provide feedback. A project cannot proceed without an approved proposal.

Your proposal must include:

- Your names and your chosen project track.
- The paper or problem you will focus on.
- Problem Statement: A clear paragraph explaining your goal.
- Methodology: A brief, step-by-step plan.
- Dataset: A link to the dataset(s) you will use.
- Evaluation Plan: Explain how you will measure success. For example, "We will compare the accuracy of our model to the accuracy reported in Table 2 of the paper."
- Track-Specific Requirement for Track 4: If you choose Track 4, your proposal must also identify the two different model families you plan to compare and state your main hypothesis.

## Project Tracks

Choose one of the following four tracks for your project.

### Track 1: Reproducibility Study

- Goal: Find a published paper where the code is not available. Implement the method from the paper and try to reproduce its key results.
- Good Project: The paper should be complex enough to be a challenge but simple enough to implement in a month. It should give enough technical detail to make your work possible.
- Main Challenge: Papers often leave out important details. Your grade depends on your effort and your analysis of why your results may differ from the paper's, not on matching their numbers exactly.

### Track 2: Novel Extension

- Goal: Choose a paper where the code is available. First, run the code and replicate a key result. Then, add a meaningful extension to the work.
- What is an extension? An extension could be applying the method to a new dataset, changing the model architecture, adding a new feature, or running an analysis the original authors did not do. A small change like tuning a single hyperparameter is not enough.
- Main Challenge: Defining an extension that is both interesting and possible to complete. You must justify why your extension is useful.

### Track 3: Replication and Rebuttal

- Goal: Find a paper with claims that seem surprising or not fully supported. Replicate the experiments to verify the claims. Your goal is to either provide stronger support for the paper's conclusion or show that its claims are incorrect.
- Good Project: This track is difficult. The paper's claims should be interesting, and its method must be clear enough for you to replicate. Your own experiments must be designed very carefully.
- Main Challenge: You may find the paper was correct after all. Success is based on the quality of your investigation, not on proving the paper wrong. You must get instructor approval for the paper you choose for this track.

### Track 4: Applied Machine Learning Investigation

- Goal: Conduct a deep investigation of a challenging applied problem. The focus is on comparing different methods and analyzing your results, not just on getting a high score.
- Mandatory Requirements: You must implement and compare at least two fundamentally different model families (e.g., a decision tree ensemble vs. a neural network). Your report must also include a dedicated error analysis section where you investigate the types of mistakes your best model makes.
- Main Challenge: Justifying your design choices at every step. Directly copying a public solution is not acceptable. You must build your own pipeline and provide a critical analysis of your model's performance that goes beyond a single accuracy score.

## Choosing a Dataset

The dataset you choose will affect your project.
- For Tracks 1, 2, and 3, you should use the same dataset as the paper you are studying. These are often standard research datasets found in places like the Hugging Face Hub. This is important for a fair comparison.
- For Track 4, you should find a problem that presents a realistic challenge. Kaggle and the UCI Machine Learning Repository are good sources. Look for a dataset that requires data cleaning, feature engineering, and model comparison. Avoid problems that have many perfect public solutions.

## Timeline and Final Deliverables

- Nov. 12: Project Proposal Due.
- Week of Dec. 1: Progress Check-in. A brief meeting with the TA to discuss your progress.
- Dec. 8: Final Submission Due.
- Dec. 10, 8 pm - 10:30 pm: Project Presentation Session.

Your final submission must include three parts:

- Final Report (6-8 pages): A report in a standard conference paper format (Abstract, Introduction, Methods, Results, Conclusion). It should clearly tell the story of your project. Upload to Canvas.
- Source Code: A link to a Git repository (like GitHub). Your code should be clean and commented. Include a README.md file with instructions on how to set up the environment and run your code.
- Pre-recorded Video Presentation (8-10 minutes): You will create a short video presenting your project, similar to a conference presentation. It must summarize your project's goal, methods, key results, and conclusions. During our scheduled final exam time, we will play each group's video. This will be followed by a live question and answer (Q&A) session with the course staff. Both team members must be present for the Q&A. Upload to Canvas.
