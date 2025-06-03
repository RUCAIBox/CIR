# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import os
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

import copy, json
from .code_excute import excute_code, extract_code, PythonExecutor, detect_code, process_string, extract_code_sole
import requests

# vllm_urls = [
#     "http://localhost:8001/generate",  # 对应 GPU 4
#     "http://localhost:8002/generate",  # 对应 GPU 5
#     "http://localhost:8003/generate",  # 对应 GPU 6
#     "http://localhost:8004/generate"   # 对应 GPU 7
# ]

# vllm_urls = [
#     "http://172.27.37.217:9001/generate",  # 对应 GPU 4
#     "http://172.27.37.217:9002/generate",  # 对应 GPU 5
#     "http://172.27.37.217:9003/generate",  # 对应 GPU 6
#     "http://172.27.37.217:9004/generate",  # 对应 GPU 7
#     "http://172.27.37.217:9005/generate",  # 对应 GPU 4
#     "http://172.27.37.217:9006/generate",  # 对应 GPU 5
#     "http://172.27.37.217:9007/generate",  # 对应 GPU 6
#     "http://172.27.37.217:9008/generate"   # 对应 GPU 7
# ]

# worker1的
# vllm_urls = [
#     "http://172.27.37.217:9011/generate",  # 对应 GPU 4
#     "http://172.27.37.217:9012/generate",  # 对应 GPU 5
#     "http://172.27.37.217:9013/generate",  # 对应 GPU 6
#     "http://172.27.37.217:9014/generate",  # 对应 GPU 7
#     "http://172.27.37.217:9015/generate",  # 对应 GPU 4
#     "http://172.27.37.217:9016/generate",  # 对应 GPU 5
#     "http://172.27.37.217:9017/generate",  # 对应 GPU 6
#     "http://172.27.37.217:9018/generate"   # 对应 GPU 7
# ]

# worker3的
vllm_urls = [
    "http://172.24.69.255:9021/generate",  # 对应 GPU 4
    "http://172.24.69.255:9022/generate",  # 对应 GPU 5
    "http://172.24.69.255:9023/generate",  # 对应 GPU 6
    "http://172.24.69.255:9024/generate",  # 对应 GPU 7
    "http://172.24.69.255:9025/generate",  # 对应 GPU 4
    "http://172.24.69.255:9026/generate",  # 对应 GPU 5
    "http://172.24.69.255:9027/generate",  # 对应 GPU 6
    "http://172.24.69.255:9028/generate"   # 对应 GPU 7
]

def get_current_gpu_id():
    """
    获取当前主程序使用的 GPU ID。
    假设主程序通过 CUDA_VISIBLE_DEVICES 环境变量绑定到特定 GPU。
    """
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_ids = list(map(int, cuda_visible_devices.split(",")))
    return gpu_ids[0]  # 返回第一个 GPU ID

# def send_request(prompt):
#     """
#     根据当前 GPU ID，选择对应的 vLLM 实例发送请求。
#     """
#     current_gpu_id = get_current_gpu_id()
#     if current_gpu_id not in [0, 1, 2, 3]:
#         raise ValueError(f"Invalid GPU ID: {current_gpu_id}. Expected one of [0, 1, 2, 3].")

#     # 计算对应的 vLLM 实例 URL
#     vllm_url = vllm_urls[current_gpu_id]
#     try:
#         response = requests.post(vllm_url, json={"prompt": prompt})
#         return response.json()
#     except Exception as e:
#         print(f"Error sending request to {vllm_url}: {e}")
#         return None

def send_request(prompts):
    """
    根据当前 GPU ID，选择对应的 vLLM 实例发送批量请求。
    
    参数:
        prompts (list): 输入提示列表，必须是一个字符串列表。
    
    返回:
        dict 或 None: 服务端返回的 JSON 数据，或在发生错误时返回 None。
    """
    # 验证输入是否为列表
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts 必须是一个字符串列表")

    # 获取当前 GPU ID
    current_gpu_id = get_current_gpu_id()
    # if current_gpu_id not in [0, 1, 2, 3]:
    if current_gpu_id not in [0, 1, 2, 3, 4, 5, 6, 7]:
        raise ValueError(f"Invalid GPU ID: {current_gpu_id}. Expected one of [0, 1, 2, 3].")

    # 计算对应的 vLLM 实例 URL
    vllm_url = vllm_urls[current_gpu_id]
    try:
        # 构造请求数据
        payload = {"prompts": prompts}  # 使用 "prompts" 字段支持批量处理

        # 发送 POST 请求
        response = requests.post(vllm_url, json=payload)

        # 打印调试信息
        print(f"Request URL: {vllm_url}")
        #print(f"Request Payload: {payload}")
        print(f"Response Status Code: {response.status_code}")
        #print(f"Response Body: {response.text}")

        # 检查响应状态码
        if response.status_code != 200:
            raise Exception(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

        # 解析并返回 JSON 数据
        return response.json()

    except requests.exceptions.RequestException as e:
        # 捕获网络请求异常
        print(f"Network error: {e}")
        return None

    except ValueError as e:
        # 捕获 JSON 解析异常或输入验证错误
        print(f"Parameter error: {e}")
        return None

    except Exception as e:
        # 捕获其他异常
        print(f"An error occurred: {e}")
        return None


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=int(os.getenv("RANK", "0")) // tensor_parallel_size,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        self.tokenizer = tokenizer
        self.exectuor = PythonExecutor()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences_origin(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=True)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    
    @torch.no_grad()
    def generate_sequences_CIR(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")
        
        #  输入的还是字符串prompt
        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        decoded_texts = [
            self.tokenizer.decode(tokens, skip_special_tokens=False)
            for tokens in idx_list
        ]
        
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        print(f"sign!!!: {is_validate}")
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            coding_sampling_params = copy.deepcopy(self.sampling_params)
            coding_sampling_params.detokenize = True
            #coding_sampling_params.stop=["User:","Human:","Assistant:"] #改一下 update0403
            print("coding Sampling_params: ", coding_sampling_params) 
            total_response_num = len(idx_list) * coding_sampling_params.n
            print('total_response_num = len(idx_list) * coding_sampling_params.n---', f'{total_response_num} = {len(idx_list)} * {coding_sampling_params.n}') # update，一定不要写self.config.n
           
            outputs = self.inference_engine.generate(
                #prompts=vllm_inputs,  # because we have already convert it to prompt token id
                prompts=decoded_texts,
                sampling_params=coding_sampling_params,
                use_tqdm=True)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            complete_data = []
            roll_candidates = []
            successful_executions_per_response = [0] * total_response_num # update0426
            # print(responses)
            # responses: list of RequestOutput [query_num, n_samples]
            index = 0
            preds_w_idx = []
            for response, prompt in zip(outputs, decoded_texts):
                #preds_w_idx = []
                for output in response.outputs:
                    pred = output.text
                    #print("******************")
                    #print(pred)
                    stop_reason = output.stop_reason
                    # print(f"stop_reason: {stop_reason}", "pred:", pred)
                    # stop_reason = output.stop_reason
                    #if (stop_reason != "```output") or (not detect_code(pred)):
                    prefix_str = "programming task below" # update: 这里区分code任务
                    if prefix_str in prompt or not detect_code(pred): # ```python ```python这种检测不出
                        complete_data.append([output, prompt, index])
                    else:
                        #preds_w_idx.append(pred, index)
                        preds_w_idx.append((process_string(pred).strip(), index, prompt))
                    index += 1
                # if len(preds_w_idx) == 0:
                #     continue
            preds = [_[0] for _ in preds_w_idx]
            temp_prompts = [_[2] for _ in preds_w_idx]
            codes = extract_code(preds)
            code_outputs, no_code_idx = excute_code(codes, self.exectuor)
            for i, (excu_result, each_data) in enumerate(zip(code_outputs, preds)):
                output, report = excu_result
                if report == "Done":
                    excu_content = output
                    successful_executions_per_response[preds_w_idx[i][1]] += 1
                else:
                    excu_content = report
                prompt_with_code = (
                    temp_prompts[i]
                    + preds[i].strip()
                    + "\n```output\n"
                    + excu_content
                    + "\n```\n"
                )
                roll_candidates.append((prompt_with_code, preds_w_idx[i][1]))
            #max_loop_num = 10
            #max_loop_num = 8
            #max_loop_num = 5
            #max_loop_num = 4
            #max_loop_num = 2
            max_loop_num = self.config.get("max_call_time", 2)
            print(f"max_call_time: {max_loop_num}")
            loop_count = 0
            continue_sampling_params = copy.deepcopy(coding_sampling_params)
            continue_sampling_params.n = 1
            print('total_response_num:', total_response_num)
            print('len of complete_data', len(complete_data))
            while len(complete_data) < total_response_num:
                loop_count += 1
                con_index = [_[1] for _ in roll_candidates]
                roll_candidates_prompt = [_[0] for _ in roll_candidates]
                # print("num_of_roll_candidate: ", len(roll_candidates_prompt))
                # print(roll_candidates_prompt)
                print('total_response_num:', total_response_num, 'continue prompts num:', len(roll_candidates_prompt))
                if len(roll_candidates_prompt) < 3:
                    print('roll_candidates', roll_candidates, roll_candidates_prompt)
                con_responses = self.inference_engine.generate(
                    prompts=roll_candidates_prompt,
                    sampling_params=continue_sampling_params,
                    use_tqdm=False,
                )
                inloop_index = 0
                roll_candidates = []
                preds_w_idx = []
                for response, prompt in zip(con_responses, roll_candidates_prompt):
                    #preds_w_idx = []
                    for output in response.outputs:
                        pred = output.text
                        # print(pred)
                        stop_reason = output.stop_reason
                        # if (
                        #     (stop_reason != "```output")
                        #     or (not detect_code(pred))
                        #     or loop_count >= max_loop_num
                        # ):
                        if (
                            loop_count >= max_loop_num
                            or (not detect_code(pred))
                        ): 
                            
                            complete_data.append(
                                [output, prompt, con_index[inloop_index]]
                            )
                        else:
                            preds_w_idx.append((process_string(pred).strip(), con_index[inloop_index], prompt))
                        inloop_index += 1
                    # preds = [_[0] for _ in preds_w_idx]
                    # if len(preds_w_idx) <= 0:
                    #     continue
                preds = [_[0] for _ in preds_w_idx]
                temp_prompts = [_[2] for _ in preds_w_idx]
                codes = extract_code(preds)
                code_outputs, no_code_idx = excute_code(codes, self.exectuor)
                for i, (excu_result, each_data) in enumerate(
                    zip(code_outputs, preds)
                ):
                    output, report = excu_result
                    if report == "Done":
                        excu_content = output
                        successful_executions_per_response[preds_w_idx[i][1]] += 1
                    else:
                        excu_content = report
        
                    prompt_with_code = (
                        temp_prompts[i]
                        + preds[i].strip()
                        + "\n```output\n"
                        + excu_content
                        + "\n```\n"
                    )
                    roll_candidates.append((prompt_with_code, preds_w_idx[i][1]))
                if loop_count >= max_loop_num:
                    break
                
            print('='*10, 'Code Injected', '='*10)
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            final_response_list = [0] * total_response_num
            final_sentence_list = [0] * total_response_num
            for each_data in complete_data:
                # print(each_data)
                ori_idx = each_data[2]
                ori_prompt_idx = ori_idx // coding_sampling_params.n
                ori_prompt = decoded_texts[ori_prompt_idx]
                output = each_data[0]
                final_prompt = each_data[1]
                
                
                final_sentence = final_prompt + output.text + self.tokenizer.eos_token
                final_response = final_sentence.replace(ori_prompt, "", 1)
                final_response_list[ori_idx] = final_response
                final_sentence_list[ori_idx] = final_sentence
            response = self.tokenizer(
                final_response_list,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.config.response_length,
            )["input_ids"]

                

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                            max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                    self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response,
                                                    eos_token=eos_token_id,
                                                    dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'pass_num' : torch.tensor(successful_executions_per_response)
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # print('vllm_rollout kwargs: ', kwargs)
        # generate_method = kwargs['generate_method']
        generate_method = self.config.get("generate_method", "naive")
        print("generate_method: ", generate_method)
        if generate_method == "CIR":
            return self.generate_sequences_CIR(prompts)
        elif generate_method == "naive":
            return self.generate_sequences_origin(prompts)
        else:
            raise NotImplementedError