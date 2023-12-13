# -*- coding: utf-8 -*- 
"""
    @Project ：quickllm 
    @File    ：triton_client.py
    @Author  ：ys
    @Time    ：2023/12/13 11:26
    请求一次，方便调试
"""

import time
from typing import Dict, List
from tritonclient.http import InferenceServerClient
from tritonclient.http import InferAsyncRequest
from tritonclient.http import InferRequestedOutput
from tritonclient.utils import InferenceServerException
from tritonclient.http import InferInput
import numpy as np
from loguru import logger
import os
import uuid


class TritonHttpClient:

    def __init__(self, url, concurrency: int, max_greenlets: int,
                 timeout=None, network_timeout=60.0, keep_shape=False, **kwargs):
        self.ctx = InferenceServerClient(url=url, concurrency=concurrency, max_greenlets=max_greenlets,
                                         network_timeout=network_timeout)
        self.ctx.is_server_live()
        self.model_config = kwargs["model_config"]
        self.outputs = kwargs["outputs"]
        self.timeout = timeout
        self.request_id = str(uuid.uuid4())

        model_info = self.ctx.get_model_repository_index()
        model_info = [ModelInfo(**_) for _ in model_info]

        for _ in model_info:
            config = self.ctx.get_model_config(_.name, _.version)
            outputs = []
            for output in config['output']:
                outputs.append(InferRequestedOutput(output['name'], binary_data=True))
            config['model_version'] = _.version
            self.model_config[_.name] = config
            self.outputs[_.name] = outputs

        self.keep_shape = keep_shape

    def _convert_to_inputs(self, name, data: Dict[str, np.ndarray]):

        inputs = []
        config = self.model_config[name]
        for input_ in config['input']:
            if input_['optional'] and input_['name'] not in data:
                continue
            dims = input_['dims']
            input_name = input_['name']
            data_type = input_['data_type'].split('_')[-1]

            input_value = data[input_name]
            if len(dims) - len(list(input_value.shape)) == -1 and not self.keep_shape:
                input_value = np.squeeze(input_value, axis=-1)

            if self.keep_shape:
                shape = input_value.shape
            else:
                shape = [a if b != -1 else a for a, b in zip(list(input_value.shape), list(dims))]
            temple = InferInput(input_name, shape, data_type)
            temple.set_data_from_numpy(input_value, binary_data=True)
            inputs.append(temple)

        return inputs

    def run(self, name, data: Dict[str, np.ndarray]):

        inputs = self._convert_to_inputs(name, data)
        rt = self.ctx.infer(model_name=name,
                            inputs=inputs,
                            model_version=self.model_config[name]['model_version'],
                            outputs=self.outputs[name],
                            request_id=self.request_id,
                            timeout=self.timeout)
        return rt
