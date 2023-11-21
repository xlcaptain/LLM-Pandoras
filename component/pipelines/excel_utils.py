import base64
import os
import queue
import re
from io import BytesIO
from subprocess import PIPE
import jupyter_client
from PIL import Image
from loguru import logger

IPYKERNEL = os.environ.get('IPYKERNEL', 'paddle')


def get_kernel():
    return CodeKernel()


def get_system_messages(filename):
    PRESET_CODE = f"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

# # 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.size'] = 18

# read data
df = pd.read_excel("{filename}")
"""
    code = PRESET_CODE + "df.info()"
    _, res = execute(code, get_kernel())

    SYSTEM_MESSAGE = [
        {
            "role": "system",
            "content": """你是一位智能AI助手，你叫"大模型"，你连接着一台电脑，但请注意不能联网。在使用Python解决任务时，你可以运行代码并得到结果，如果运行结果有错误，你需要尽可能对代码进行改进。
    你主要基于 pandas 库来执行代码并输出结果，以此来回答用户的相关问题。现在已经通过 pandas 正确加载了数据文件并创建了初始数据框 pd.DataFrame，其名称为 df。"""
        },
        {
            "role": "user",
            "content": "查看一下数据框 df 的具体信息"
        },
        {
            "role": "assistant",
            "content": f""" interpreter\n要查看数据框 df 的具体信息，您可以使用 `info()` 函数。

    ```python
    {code}
    ```
    """
        },
        {
            "role": "function",
            "content": res,
        },
        {
            "role": "assistant",
            "content": f"根据查询结果，该数据框的具体信息为：\n{res}"
        },
    ]

    code = PRESET_CODE + "df.head(2)"
    _, res = execute(code, get_kernel())
    SYSTEM_MESSAGE.extend(
        [
            {
                "role": "user",
                "content": "查看数据的前两行"
            },
            {
                "role": "assistant",
                "content": f""" interpreter\n要查看数据框 df 的前两行，您可以使用 `head()` 函数。

            ```python
            {code}
            ```
            """
            },
            {
                "role": "function",
                "content": res,
            },
            {
                "role": "assistant",
                "content": f"根据查询结果，该数据框的前两行为：\n{res}"
            },
        ]
    )
    return SYSTEM_MESSAGE, PRESET_CODE


class CodeKernel:
    def __init__(
            self,
            kernel_name='kernel',
            kernel_id=None,
            kernel_config_path="",
            python_path=None,
            ipython_path=None,
            init_file_path="./startup.py",
            verbose=1,
    ):

        self.kernel_name = kernel_name
        self.kernel_id = kernel_id
        self.kernel_config_path = kernel_config_path
        self.python_path = python_path
        self.ipython_path = ipython_path
        self.init_file_path = init_file_path
        self.verbose = verbose

        if python_path is None and ipython_path is None:
            env = None
        else:
            env = {"PATH": self.python_path + ":$PATH", "PYTHONPATH": self.python_path}

        # Initialize the backend kernel
        self.kernel_manager = jupyter_client.KernelManager(
            kernel_name=IPYKERNEL,
            connection_file=self.kernel_config_path,
            exec_files=[self.init_file_path],
            env=env,
        )
        if self.kernel_config_path:
            self.kernel_manager.load_connection_file()
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            logger.info("Backend kernel started with the configuration: {}".format(
                self.kernel_config_path))
        else:
            self.kernel_manager.start_kernel(stdout=PIPE, stderr=PIPE)
            logger.info("Backend kernel started with the configuration: {}".format(
                self.kernel_manager.connection_file))

        if verbose:
            logger.info(self.kernel_manager.get_connection_info())

        # Initialize the code kernel
        self.kernel = self.kernel_manager.blocking_client()
        # self.kernel.load_connection_file()
        self.kernel.start_channels()
        logger.info("Code kernel started!")

    def execute(self, code):
        self.kernel.execute(code)
        try:
            shell_msg = self.kernel.get_shell_msg(timeout=30)
            io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']
            while True:
                msg_out = io_msg_content
                try:
                    io_msg_content = self.kernel.get_iopub_msg(timeout=30)['content']
                    if 'execution_state' in io_msg_content and io_msg_content['execution_state'] == 'idle':
                        break
                except queue.Empty:
                    break
            return shell_msg, msg_out
        except Exception as e:
            logger.error(e)
            return None

    def execute_interactive(self, code, verbose=False):
        shell_msg = self.kernel.execute_interactive(code)
        if shell_msg is queue.Empty:
            if verbose:
                logger.warning("Timeout waiting for shell message.")
        self.check_msg(shell_msg, verbose=verbose)
        return shell_msg

    def inspect(self, code, verbose=False):
        _ = self.kernel.inspect(code)
        shell_msg = self.kernel.get_shell_msg(timeout=30)
        if shell_msg is queue.Empty:
            if verbose:
                logger.warning("Timeout waiting for shell message.")
        self.check_msg(shell_msg, verbose=verbose)
        return shell_msg

    def get_error_msg(self, msg, verbose=False):
        if msg['content']['status'] == 'error':
            try:
                error_msg = msg['content']['traceback']
            except:
                try:
                    error_msg = msg['content']['traceback'][-1].strip()
                except:
                    error_msg = "Traceback Error"
            if verbose:
                logger.error("Error: ", error_msg)
            return error_msg
        return None

    def check_msg(self, msg, verbose=False):
        status = msg['content']['status']
        if status == 'ok':
            if verbose:
                logger.success("Execution succeeded.")
        elif status == 'error':
            for line in msg['content']['traceback']:
                if verbose:
                    logger.error(line)

    def shutdown(self):
        # Shutdown the backend kernel
        self.kernel_manager.shutdown_kernel()
        logger.info("Backend kernel shutdown.")
        # Shutdown the code kernel
        self.kernel.shutdown()
        logger.info("Code kernel shutdown.")

    def restart(self):
        # Restart the backend kernel
        self.kernel_manager.restart_kernel()

    def interrupt(self):
        # Interrupt the backend kernel
        self.kernel_manager.interrupt_kernel()

    def is_alive(self):
        return self.kernel.is_alive()


def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def clean_ansi_codes(input_string):
    ansi_escape = re.compile(r'(\x9B|\x1B\[|\u001b\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', input_string)


def execute(code, kernel: CodeKernel):
    res = ""
    res_type = None
    code = code.replace("<|observation|>", "")
    code = code.replace("<|assistant|>interpreter", "")
    code = code.replace("<|assistant|>", "")
    code = code.replace("<|user|>", "")
    code = code.replace("<|system|>", "")
    msg, output = kernel.execute(code)

    if msg['metadata']['status'] == "timeout":
        return res_type, 'Timed out'
    elif msg['metadata']['status'] == 'error':
        return res_type, clean_ansi_codes('\n'.join(kernel.get_error_msg(msg, verbose=True)))

    if 'text' in output:
        res_type = "text"
        res = output['text']
    elif 'data' in output:
        for key in output['data']:
            if 'text/plain' in key:
                res_type = "text"
                res = output['data'][key]
            elif 'image/png' in key:
                res_type = "image"
                res = output['data'][key]
                break

    if res_type == "image":
        return res_type, b64_2_img(res)
    elif res_type == "text" or res_type == "traceback":
        res = res
    return res_type, res


def extract_code(text: str) -> str:
    pattern = r'```([^\n]*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1][1]


def postprocess_text(text: str) -> str:
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    return text.strip()
