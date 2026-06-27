# Simulation of METR Time Horizon task-level results for GPT-5.6 Sol

In their latest [predeployment evaluation](https://metr.org/blog/2026-06-26-gpt-5-6-sol/), preformed for OpenAI on GPT-5.6 Sol, METR have regressed from no longer publishing the raw run data (as with their last three Time Horizon estimates) to not even publishing the n_runs and n_successes.

This script simulates plausible task-level results that fit with what METR did publish:
* 11.3h 50% Time Horizon
* 270hrs 50% Time Horizon if there are cheating attempts that are counted as successes
* 71hrs 50% Time Horizon if the cheating attempt runs are removed from the dataset

It makes the following assumptions:
* That there are some tasks for which the type of cheating observed by METR is not possible
* That GPT-5.6 Sol would not score below GPT-5.4 on any tasks
* That GPT-5.6 Sol would not score above Claude Mythos on any tasks, unless GPT-5.4 already did
* That METR actually performed 6 runs of each task (this is the assumption that I am least certain about)

The [synthetic data herein](https://github.com/BoxoMcFoxo/metr-time-horizon-gpt-5_6-sol-simulation/blob/main/task_results_1_1_gpt_5_6_sol_simulated.yaml) can be used as a placeholder by anyone performing task-level adjustments to METR's data (such as augmented baseline times, different task grouping schema, or ablations).

Should METR ever deign to release the actual figures, you should of course refer to those instead.they
