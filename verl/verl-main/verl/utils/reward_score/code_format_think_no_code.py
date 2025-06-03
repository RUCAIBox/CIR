from math_verify import parse, verify
import re
# 预编译所有正则（提升重复调用性能）注意是\s*
STRUCTURE_PATTERN = re.compile(
    r"^<think>.*?</think>\s*<answer>.*?</answer>$", 
    flags=re.DOTALL
)
TAG_PATTERN = re.compile(
    r"<(/?(?:think|answer))>"
)


def check_response(text: str) -> bool:
    # Qwen的eos去掉，不然无论如何匹配不上
    text = text.replace("<|endoftext|>","")
    # 阶段1：快速失败检查
    if len(text) < 23:
        print("format false (min length)")
        return False

    # 阶段2：增强型标签验证
    tag_stack = []
    tag_counter = {'think': 0, 'answer': 0}  # 标签出现计数器
    
    for match in TAG_PATTERN.finditer(text):
        tag = match.group(1)
        
        # 处理开始标签
        if tag in ['think', 'answer']:
            # 检查标签重复
            if tag_counter[tag] >= 1:
                print(f"duplicate <{tag}>")
                return False
            tag_counter[tag] += 1
            tag_stack.append(tag)
            
        # 处理结束标签
        elif tag in ['/think', '/answer']:
            base_tag = tag[1:]
            # 检查标签闭合顺序
            if not tag_stack or tag_stack[-1] != base_tag:
                print(f"unclosed {tag}")
                return False
            tag_stack.pop()
            
    # 最终完整性检查
    if tag_stack:
        print(f"unclosed {tag_stack[-1]}")
        return False

    # 阶段3：结构验证
    if not STRUCTURE_PATTERN.fullmatch(text):
        print("format false")
        return False

    # 阶段4：内容非空验证
    # think_content = text.split('<think>', 1)[1].split('</think>', 1)[0].strip()
    # answer_content = text.split('<answer>', 1)[1].split('</answer>', 1)[0].strip()
    
    # if not think_content or not answer_content:
    #     print("empty content")
    #     return False

    # return True
    def get_content(tag: str) -> str:
        s = text.find(f"<{tag}>") + len(tag)+2
        e = text.find(f"</{tag}>", s)
        return text[s:e].strip() if s != -1 and e != -1 else ""

    think_content = get_content("think")
    answer_content = get_content("answer")
    
    if not think_content or not answer_content:
        print("empty content")
        return False
    
    # if (not validate_code_blocks(think_content)) or (validate_code_blocks(answer_content)):
    #     print("invalid code format")
    #     return False
    
    return True
    
    
def extract_answer_math(s):
    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, s)
    matches = list(match)
    if matches:
        ans = matches[-1].group(1).strip()
    else:
        return ""
    
    ans = ans.split("boxed")
    if len(ans) == 1:
        return ans[0]
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    return a


def compute_score(solution_str, ground_truth) -> float:
    solution_str = solution_str.split("Assistant:")[-1].strip()
    solution_str = solution_str.replace("<|endoftext|>","")
    solution_str = "<think>\n" + solution_str # update for 030
    pred_ans = extract_answer_math(solution_str)
    score = 0.0
    try:
        if verify(parse("$" + pred_ans + "$"), parse("$" + ground_truth + "$")):
            score = 1.0
        else:
            score = 0.0
    except Exception as e:
        print(e)
        score = 0.0
        
    format_flag = check_response(solution_str)
    
    if not format_flag:
        score = 0.0
    else:
        if not score:
            score = 0.0
        else:
            score = 1.0
            
    return score
    
    


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
