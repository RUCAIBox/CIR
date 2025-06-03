# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

from math_verify import parse, verify


def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    # if 'Reach max function call limit' in solution_str:
    #     return 0.0
    #prefix_str = "Please provide a solution to the following problem by integrating natural language reasoning with Python codes. Begin by explaining your thought process step by step, and then implement the solution in Python. Ensure the code is clear, well-documented, and follows best practices. Finally, present the final answer enclosed within \\boxed{} for clarity.\n" 
    prefix_str = "Please solve the following problem step by step. During your reasoning process, if needed, you can choose to write python code to enhance your reasoning. The code executor will run your code and provide the execution results back to you to support your reasoning process. Please put the final answer within \\boxed{}.\n"
    #prefix_str = "<|im_end|>\n<|im_start|>assistant\n"
    solution_str = solution_str.split(prefix_str)[-1]

    # update 0315: 乱码惩罚，针对Qwen-Math. 严格惩罚 
    '''
    if "!!!" in solution_str:
        return -2.0
    '''
    # update: remove format reward
    # if not validate_code_blocks(solution_str):
    #     return 0.0

    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if verify(parse("$" + ground_truth + "$"), parse("$" + answer + "$")):
                retval = 1.0
            else:
                retval = 0.0


    except Exception as e:
        print(e)
        retval = 0.0
    if 'boxed' not in solution_str:
        retval = 0.0
    return retval


"""
def compute_score(solution_str, ground_truth) -> float:
    retval = -1.0
    # if 'Reach max function call limit' in solution_str:
    #     return 0.0
    solution_str = solution_str.split('<|im_start|>assistant\n')[-1]

    # update 0315: 乱码惩罚，针对Qwen-Math. 严格惩罚 
    '''
    if "!!!" in solution_str:
        return -2.0
    '''
    # update: remove format reward
    # if not validate_code_blocks(solution_str):
    #     return 0.0

    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if verify(parse("$" + ground_truth + "$"), parse("$" + answer + "$")):
                retval = 1.0
            else:
                retval = -1.0


    except Exception as e:
        print(e)
        retval = -1.0
    if 'boxed' not in solution_str:
        retval = -1.0
    return retval
"""


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string



def validate_code_blocks(s):
    code_blocks = []
    i = 0
    n = len(s)
    while i < n:
        if s[i:i+3] == '```':
            i += 3
            lang = []
            while i < n and s[i] not in ('\n', '\r', ' '):
                lang.append(s[i])
                i += 1
            lang = ''.join(lang).strip()
            if lang:
                close_pos = s.find('```', i)
                if close_pos == -1:
                    return False
                code_blocks.append(lang)
                i = close_pos + 3
            else:
                return False
        else:
            i += 1

    # 检查每个python必须紧跟output，且成对出现
    total_python = 0
    total_output = 0
    i = 0
    while i < len(code_blocks):
        if code_blocks[i] == 'python':
            total_python += 1
            if i + 1 >= len(code_blocks) or code_blocks[i+1] != 'output':
                return False
            total_output += 1
            i += 2  # 跳过已配对的output
        elif code_blocks[i] == 'output':
            # 单独存在的output无效
            return False
        else:
            # 其他类型代码块跳过
            i += 1

    return total_python == total_output and total_python > 0



if __name__ == '__main__':
    # 更新后的测试用例
    test_cases = [
        ("```python\ncode\n```\n```output\nout\n```", True),          # 正确配对
        ("```python\ncode1\n```\n```python\ncode2\n```\n```output\nout\n```", False),  # 连续python
        ("```output\nout\n```\n```python\ncode\n```", False),         # 顺序颠倒
        ("```python\ncode\n```\n```output\nout", False),              # 未闭合
        ("```python\ncode\n```\n```output\nout1\n```\n```python\ncode2\n```\n```output\nout2\n```", True),  # 多对
        ("```bash\necho\n```\n```python\ncode\n```\n```output\nout\n```", True),       # 包含其他类型
        ("```python\ncode\n```\n```bash\nscript\n```\n```output\nout\n```", False),    # 中间有其他代码块
        (r"""
          18    \]
  19 
  20 2. **Calculate the number of ways Jen can win a prize:**
  21    - Jen wins a prize if at least 2 of her numbers are among the 4 randomly chosen numbers.
  22    - We need to consider the cases where she has exactly 2, 3, or 4 of her numbers among the 4 chosen numbers.
  23 
  24 3. **Calculate the number of ways Jen can win the grand prize:**
  25    - Jen wins the grand prize if all 4 of her numbers are among the 4 randomly chosen numbers.
  26    - There is only 1 way for this to happen.
  27 
  28 4. **Calculate the probability of winning the grand prize given that she won a prize:**
  29    - This is the number of ways to win the grand prize divided by the number of ways to win a prize.
  30 
  31 Let's implement this in Python using sympy.
  32 
```python
  34 import sympy as sp
  35 from sympy import binomial
  36 
  37 # Total number of ways to choose 4 numbers from 10
  38 total_ways = binomial(10, 4)
  39 
  40 # Number of ways to win a prize
  41 # Case 1: Exactly 2 of her numbers are among the 4 chosen numbers
  42 ways_2_correct = binomial(4, 2) * binomial(6, 2)
  43 # Case 2: Exactly 3 of her numbers are among the 4 chosen numbers
  44 ways_3_correct = binomial(4, 3) * binomial(6, 1)
  45 # Case 3: All 4 of her numbers are among the 4 chosen numbers
  46 ways_4_correct = binomial(4, 4) * binomial(6, 0)
  47 
  48 # Total number of ways to win a prize
  49 ways_to_win_prize = ways_2_correct + ways_3_correct + ways_4_correct
  50 
  51 # Number of ways to win the grand prize
  52 ways_to_win_grand_prize = 1
  53 
  54 # Probability of winning the grand prize given that she won a prize
  55 probability_grand_given_prize = ways_to_win_grand_prize / ways_to_win_prize
  56 
  57 # Simplify the fraction
  58 probability_grand_given_prize_simplified = sp.Rational(ways_to_win_grand_prize, ways_to_win_prize)
  59 
  60 # Extract m and n
  61 m, n = probability_grand_given_prize_simplified.as_numer_denom()
  62 
  63 # Calculate m + n
  64 result = m + n
  65 
  66 print(result)
```
```output
2
```""", True)
    ]

    for i, (test_input, expected) in enumerate(test_cases):
        result = validate_code_blocks(test_input)
        assert result == expected, f"Test case {i+1} failed: {test_input}"
    print("All test cases passed!")
