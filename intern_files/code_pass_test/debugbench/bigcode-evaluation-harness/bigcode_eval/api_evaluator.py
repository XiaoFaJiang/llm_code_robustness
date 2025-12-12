import os
import copy
import json
import inspect
import warnings

from bigcode_eval import tasks
from bigcode_eval.api_utils import _make_instruction_prompt
from bigcode_eval.api_client import HttpAPIServer
from bigcode_eval.filewriter import write_save_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class APIEvaluator:
    def __init__(self, model_name, args, global_args):
        self.model_name = model_name
        self.args = args
        self.global_args = global_args
        # setup server
        self.server = HttpAPIServer(model_name=model_name, global_args=global_args)

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution
        

    def generate_text(self, task_name, global_args):
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # n_tasks = self.args.limit if self.args.limit else len(dataset)
        n_tasks = min(self.args.limit, len(dataset)) if self.args.limit else len(dataset)
        references = [task.get_reference(dataset[i]) for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(self.args.limit_start, self.args.limit_start+n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        prompts = []
        generations = []
        gen_kwargs = {
            "temperature": global_args["temperature"],
            "top_k": global_args["top_k"],
            "top_p": global_args["top_p"],
            "do_sample": global_args["do_sample"],
            "max_length": global_args["max_new_tokens"],
        }

        for sample in range(self.args.limit_start, self.args.limit_start+n_tasks):
            prompts_, generations_ = [], []
            for _ in range(self.args.n_samples):
                prompt_contents = task.get_prompt(dataset[sample])
                if isinstance(prompt_contents, str):
                    # Normal code completion mode
                    prompt = self.args.prefix + prompt_contents + self.args.suffix

                elif isinstance(prompt_contents, dict):
                    if set(prompt_contents.keys()) == {"prefix", "middle", "suffix", "prompt"}:
                        # Infilling mode
                        prompt = prompt_contents["prompt"]
                        # prompt = _make_infill_prompt(
                        #     **prompt_contents, preprefix=self.args.prefix
                        # )
                    elif set(prompt_contents.keys()) == {"instruction", "context"}:
                        # Instruction-tuning mode
                        prompt = _make_instruction_prompt(
                            **prompt_contents, prefix=self.args.prefix, suffix=self.args.suffix
                        )
                else:
                    raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")

                prompts_.append(prompt)
                raw_request = {}
                raw_request["prompt"] = prompt
                completions = self.server.serve_request(raw_request, global_args, **gen_kwargs)
                generation = completions["completions"][0]["text"]
                if (not self.args.model_type == "causal_base") and self.args.suffix is not None and len(self.args.suffix) > 0:
                    prompt = prompt[: -len(self.args.suffix)]
                prompt = prompt[len(self.args.prefix) :]
                generation = task.postprocess_generation(prompt+generation, sample+self.args.limit_start)
                generations_.append(generation)
            prompts.append(prompts_)
            generations.append(generations_)

        if len(generations[0]) > self.args.n_samples:
            generations = [l[: self.args.n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return prompts, generations, references

    def evaluate(self, task_name, global_args):
        task = tasks.get_task(task_name, self.args)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        prompts, generations, references = self.generate_text(task_name, global_args)

        if not self.args.load_generations_path:
            if self.args.save_generations:
                save_generations_dir = self.args.save_generations_path
                save_generations_path = f"{save_generations_dir}/{task_name}"
                write_save_generations(prompts, generations, references, save_generations_path)

        # make sure tokenizer plays nice with multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if self.allow_code_execution and task.requires_execution:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        print("Evaluating generations...")
        metrics, cases, stats = task.process_results(generations, references)
        return metrics, cases, stats
