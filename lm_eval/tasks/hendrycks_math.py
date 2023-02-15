"""
Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/pdf/2103.03874.pdf

Math is a dataset of 12,500 challenging competition mathematics problems. Each
problem in Math has a full step-by-step solution which can be used to teach
models to generate answer derivations and explanations.

Homepage: https://github.com/hendrycks/math
"""
import inspect
import signal
import re
import sympy
import numpy as np
from contextlib import contextmanager
import lm_eval.datasets.hendrycks_math.hendrycks_math
from lm_eval.metrics import mean
from lm_eval.base import Task, rf


_CITATION = """
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the Math Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""

# Simple python eval calculator
# taken from
# https://github.com/openai/grade-school-math/blob/master/grade_school_math/calculator.py
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None

# MATH dataset answer normalization, taken from https://arxiv.org/pdf/2206.14858.pdf
SUBSTITUTIONS = [
    ('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''),
    (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
    ('\\text{and}', ','), ('\\text{m}', '\\text{}')
]
REMOVED_EXPRESSIONS = [
    'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
    'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet',
    'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
    'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
    '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2',
    '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}',
    r'^\circ', r'^{\circ}', r'\;', r',\!', '{,}', '"', '\\dots'
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split('=')[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, '')

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
    final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
    final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

    # Normalize shorthand TeX:
    # \fracab -> \frac{a}{b}
    # \frac{abc}{bef} -> \frac{abc}{bef}
    # \fracabc -> \frac{a}{b}c
    # \sqrta -> \sqrt{a}
    # \sqrtab -> sqrt{a}b
    final_answer = re.sub(
    r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
    final_answer = re.sub(
    r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
    final_answer = final_answer.replace('$', '')

    # Normalize 100,000 -> 100000
    if final_answer.replace(',', '').isdigit():
        final_answer = final_answer.replace(',', '')

    return final_answer

# Sympy-based calculator for checking if two entities are the same, taken from https://arxiv.org/pdf/2206.14858.pdf
def numeric_equality(n1, n2, threshold=0.01):
    if n1 is None or n2 is None:
        return False
    if np.isclose(n1,0) or np.isclose(n2,0) or np.isclose(n1-n2,0):
        return np.abs(n1-n2) < threshold * (n1+n2)/2
    else:
        return np.isclose(n1, n2)
    
def symbolic_equality(x,y):
    if x is None or y is None:
        return False
    else:
        try:
            return sympy.simplify(x-y) == 0
        except:
            return False
        
def normalize_symbolic_equation(s: Optional[str]):
    if not isinstance(s, str):
        return None
    if s.startswith('\\['):
        s = s[2:]
    if s.endswith('\\]'):
        s = s[:-2]
    s = s.replace('\\left(', '(')
    s = s.replace('\\right)', ')')
    s = s.replace('\\\\', '\\')
    if s.startswith('$') or s.endswith('$'):
        s = s.strip('$')
    try:
        maybe_expression = sympy.parsing.latex.parse_latex(s)
        if not isinstance(maybe_expression, sympy.core.relational.Equality):
            # we have equation, not expression
            return None
        else:
            return maybe_expression
    except:
        return None

def normalize_symbolic_expression(s: Optional[str]):
    if not isinstance(s, str):
        return None
    if s.startswith('\\['):
        s = s[2:]
    if s.endswith('\\]'):
        s = s[:-2]
    s = s.replace('\\left(', '(')
    s = s.replace('\\right)', ')')
    s = s.replace('\\\\', '\\')
    if s.startswith('$') or s.endswith('$'):
        s = s.strip('$')
    try:
        maybe_expression = sympy.parsing.latex.parse_latex(s)
        if isinstance(maybe_expression, sympy.core.relational.Equality):
            # we have equation, not expression
            return None
        if isinstance(maybe_expression, sympy.logic.boolalg.BooleanFalse):
            return None
        else:
            return maybe_expression
    except:
        return None

class Math(Task):
    DATASET_PATH = inspect.getfile(lm_eval.datasets.hendrycks_math.hendrycks_math)
    DATASET_NAME = None
    MAJORITY_VOTING = "majority_voting"
    SAMPLING_TEMPERATURE = "sampling_temperature"
    EVAL_BATCH_SIZE = "eval_batch_size"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        doc["answer"] = self.remove_boxed(self.last_boxed_only_string(doc["solution"]))
        return doc

    def doc_to_text(self, doc):
        return "Problem: " + doc["problem"] + "\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["problem"]

    def doc_to_target(self, doc):
        return " " + doc["solution"]

    def parse_description(self, description):
        """description is a string with comma-separated key=value tuples
        e.g.: 
        "majority_voting=32,sampling_temperature=0.3,eval_batch_size=8"
        """
        parsed_dict = {}
        for term in description.split(","):
            if not term.strip():
                continue
            key, value = term.split("=")
            parsed_dict[key] = value
        return parsed_dict

    def construct_requests(self, doc, ctx, description=""):
        if not description.strip():
            return rf.generate(ctx, ["\n"])
        
        parsed_description = self.parse_description(description=description)
        majority_voting_value = int(parsed_description.get(self.MAJORITY_VOTING, 1))
        sampling_temperature_value = float(parsed_description.get(self.SAMPLING_TEMPERATURE, 1.0))
        eval_batch_size = parsed_description.get(self.EVAL_BATCH_SIZE, None)
        eval_batch_size = int(eval_batch_size) if isinstance(eval_batch_size, str) else eval_batch_size
        return rf.generate(ctx, ["\n"], 
            majority_voting_value, sampling_temperature_value, eval_batch_size)
    
    def get_pure_answer(self, candidate):
        indices = [pos for pos, char in enumerate(candidate) if char == "$"]
        if len(indices) <= 1:
            return candidate
        return candidate[indices[0] + 1 : indices[-1]]

    def majority_vote(self, candidates):
        answers = []
        for candidate in candidates:
            answer = self.get_pure_answer(candidate)
            try:
                answer = self.remove_boxed(self.last_boxed_only_string(answer))
            except:
                answer = None
            answers.append(answer)
        
        answer_votes = {}
        for answer in answers:
            answer_votes[answer] = answer_votes.get(answer, 0) + 1

        max_vote = 0
        elected = None
        for answer, vote in answer_votes.items():
            if vote > max_vote and answer is not None:
                elected = answer
                max_vote = vote
        return elected

    def process_results(self, doc, results, description=""):
        retval = 0

        assert isinstance(description, str)
        if description == "":
            last_box_content = self.last_boxed_only_string(results[0])
            answer = self.get_pure_answer(self.remove_boxed(last_box_content)) if last_box_content is not None else self.get_pure_answer(results[0])
        elif self.MAJORITY_VOTING in self.parse_description(description):
            answer = self.majority_vote(results[0])
        else:
            raise AssertionError

        if self.is_equiv(
            answer, self.remove_boxed(self.last_boxed_only_string(doc["solution"]))
        ):
            retval = 1

        return {"acc": retval}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}

    def is_equiv(self, str1, str2, verbose=False):
        if str1 is None and str2 is None:
            print("WARNING: Both None")
            return True
        if str1 is None or str2 is None:
            return False

        try:
            ss1 = self.strip_string(str1)
            ss2 = self.strip_string(str2)
            if verbose:
                print(ss1, ss2)
            return ss1 == ss2
        except Exception:
            return str1 == str2

    def remove_boxed(self, s):
        if "\\boxed " in s:
            left = "\\boxed "
            assert s[: len(left)] == left
            return s[len(left) :]

        left = "\\boxed{"

        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]

    def last_boxed_only_string(self, string):

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
            retval = string[idx : right_brace_idx + 1]

        return retval

    def fix_fracs(self, string):
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

    def fix_a_slash_b(self, string):
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

    def remove_right_units(self, string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def fix_sqrt(self, string):
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

    class NotEqual:
        def __eq__(self, other):
            return False

    def strip_string(self, string):
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
        string = self.remove_right_units(string)

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
        string = self.fix_sqrt(string)

        # remove spaces
        string = string.replace(" ", "")

        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
        string = self.fix_fracs(string)

        # manually change 0.5 --> \frac{1}{2}
        if string == "0.5":
            string = "\\frac{1}{2}"

        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        string = self.fix_a_slash_b(string)

        return string


class MathAlgebra(Math):
    VERSION = 1
    DATASET_NAME = "algebra"


class MathCountingAndProbability(Math):
    VERSION = 1
    DATASET_NAME = "counting_and_probability"


class MathGeometry(Math):
    VERSION = 1
    DATASET_NAME = "geometry"


class MathIntermediateAlgebra(Math):
    VERSION = 1
    DATASET_NAME = "intermediate_algebra"


class MathNumberTheory(Math):
    VERSION = 1
    DATASET_NAME = "number_theory"


class MathPrealgebra(Math):
    VERSION = 1
    DATASET_NAME = "prealgebra"


class MathPrecalculus(Math):
    VERSION = 1
    DATASET_NAME = "precalculus"
