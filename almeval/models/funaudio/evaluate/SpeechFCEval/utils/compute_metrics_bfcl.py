# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import ast
import json
import codecs
import xlsxwriter
from tqdm import tqdm
from collections import Counter


def parse_json_line(file_path):
    data_list = []
    with codecs.open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            data_item = json.loads(line)
            data_list.append(data_item)
    return data_list


def save_json_line(data_list, file_path):
    with codecs.open(file_path, "w", encoding="utf-8") as fw:
        for data_item in data_list:
            line = json.dumps(data_item, ensure_ascii=False)
            fw.write(f"{line}\n")


def parse_json(file_path):
    with codecs.open(file_path, "r", encoding="utf-8") as fr:
        data_list = json.load(fr)
    return data_list


def save_json(data_list, file_path):
    with codecs.open(file_path, "w", encoding="utf-8") as fw:
        json.dump(data_list, fw, ensure_ascii=False, indent=2)


#### Constants ####
PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
    "number": int,
    "object": dict,
}

# This is the list of types that we need to recursively check its values
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]
NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]



#### Helper functions for AST ####
def find_description(func_descriptions, name):
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    else:
        # it is a dict, there is only one function
        return func_descriptions


def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


# def convert_func_name(function_name, model_name: str):
#     model_name_escaped = model_name.replace("_", "/")
#     if "." in function_name:
#         if MODEL_CONFIG_MAPPING[model_name_escaped].underscore_to_dot:
#             # OAI does not support "." in the function name so we replace it with "_". ^[a-zA-Z0-9_-]{1,64}$ is the regex for the name.
#             # This happens for OpenAI, Mistral, and Google models
#             return re.sub(r"\.", "_", function_name)
#     return function_name


def convert_func_name(function_name):
    # OAI does not support "." in the function name so we replace it with "_". ^[a-zA-Z0-9_-]{1,64}$ is the regex for the name.
    # This happens for OpenAI, Mistral, and Google models
    if "." in function_name:
        return re.sub(r"\.", "_", function_name)
    return function_name


def type_checker(
    param: str,
    value,
    possible_answer: list,
    expected_type_description: str,
    expected_type_converted,
    nested_type_converted,
):
    # NOTE: This type checker only supports nested type checking for one level deep.
    # We didn't implement recursive type checking for nested types, as it's not needed for the current use case and it's very complex.

    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }

    is_variable = False
    # check for the case where a variable is used instead of a actual value.
    # use the type in possible_answer as the expected type
    possible_answer_type = get_possible_answer_type(possible_answer)
    # if possible_answer only contains optional parameters, we can't determine the type
    if possible_answer_type != None:
        # we are being precise here.
        # in fact, possible_answer_type should always be string, as that's how we treat varibale in possible_answer
        if possible_answer_type != expected_type_converted:
            is_variable = True

    # value is the same type as in function description
    if type(value) == expected_type_converted:
        # We don't need to do recursive check for simple types
        if nested_type_converted == None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True  # Each parameter should match to at least one possible answer type.
                # Here, we assume that each item should be the same type. We could also relax it.
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break

                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. Expected outer type {expected_type_description} with inner type {str(nested_type_converted)}. Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"

    # value is not as expected, check for the case where a variable is used instead of a actual value
    # use the type in possible_answer as the expected type
    possible_answer_type = get_possible_answer_type(possible_answer)
    # if possible_answer only contains optional parameters, we can't determine the type
    if possible_answer_type != None:
        # we are being precise here.
        # in fact, possible_answer_type should always be string, as that's how we treat varibale in possible_answer
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. Expected type {expected_type_description}, got {type(value).__name__}. Parameter value: {repr(value)}."
    )
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str):
    """
    This function standardizes the string by removing all the spaces, ",./-_*^" punctuation, and converting it to lowercase
    It will also convert all the single quotes to double quotes
    This is used to compare the model output with the possible answers
    We don't want to punish model for answer like April 1, 2024 vs April 1,2024, vs April 1 2024
    """
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def string_checker(param: str, model_output: str, possible_answer: list):
    standardize_possible_answer = []
    standardize_model_output = standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }

    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list):
    # Convert the tuple to a list

    standardize_model_output = list(model_output)

    # If the element in the list is a string, we need to standardize it
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = standardize_string(model_output[i])

    standardize_possible_answer = []
    # We also need to standardize the possible answers
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) == str:
                standardize_possible_answer[i].append(
                    standardize_string(possible_answer[i][j])
                )
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }

    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list):
    # This function works for simple dictionaries, but not dictionaries with nested dictionaries.
    # The current dataset only contains simple dictionaries, so this is sufficient.

    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):

        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}

        flag = True

        possible_answer = possible_answers[i]
        # possible_answer is a single dictionary

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = value
            # If the value is a string, we need to standardize it
            if type(value) == str:
                standardize_value = standardize_string(value)

            # We also need to standardize the possible answers if they are string
            standardize_possible_answer = []
            for i in range(len(possible_answer[key])):
                if type(possible_answer[key][i]) == str:
                    standardize_possible_answer.append(
                        standardize_string(possible_answer[key][i])
                    )
                else:
                    standardize_possible_answer.append(possible_answer[key][i])

            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(key)}: {repr(value)}. Expected one of {standardize_possible_answer}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list):
    # This function takes in a list of dictionaries and checks if each dictionary is valid
    # The order of the dictionaries in the list must match the order of the possible answers

    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for answer_index in range(len(possible_answers)):
        flag = True  # True means so far, all dictionaries are valid

        # Only proceed if the number of dictionaries in the list matches the number of dictionaries in the possible answers
        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = dict_checker(
                param,
                model_output[dict_index],
                [possible_answers[answer_index][dict_index]],
            )
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}

    return result


#### Evaluation function ####

def simple_function_checker(
    func_description: dict,
    model_output: dict,
    possible_answer: dict,
    language: str,
    model_name: str,
):
    possible_answer = list(possible_answer.values())[0]
    # Extract function name and parameters details
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]

    # Initialize a result dictionary
    result = {
        "valid": True,
        "error": [],
        "error_type": "",  # simple_function_checker:unclear
    }

    if "gpt-4o" in model_name:
        func_name = convert_func_name(func_name)

    # Check if function name matches. Function name 出错
    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(
            f"Function name {repr(func_name)} not found in model output."
        )
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    # Check for required parameters in model output. 缺失必要参数
    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    # Validate types and values for each parameter in model output
    for param, value in model_params.items():
        # 参数 key 不合法
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]  # This is a string

        if expected_type_description == "object":
            continue

        is_variable = False
        nested_type_converted = None

        if language == "python":
            expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
            if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
                nested_type = param_details[param]["items"]["type"]

                # if nested_type == "object":
                #     continue

                nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        else:
            raise ValueError(f"Unsupported language: {language}")

        # We convert all tuple value to list when the expected type is tuple.
        # The conversion is necessary because any tuple in the possible answer would become a list after being processed through json.dump() and json.load().
        # This does introduce some false positive (eg, when the model provides a list value instead of tuple). We hope to find a better solution in the future.
        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)

        # Allow python auto conversion from int to float
        if (
            language == "python"
            and expected_type_description == "float"
            and type(value) == int
        ):
            value = float(value)

        # 参数 type 格式检查
        # Type checking
        # In fact, we only check for Python here.
        # Type check for other languages are handled by the type converter, and so their value (after conversion) is always correct.
        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        # It doesn't make sense to special handle dictionaries and list of dictionaries if the value is a variable.
        # We can just treat the variable as a string and use the normal flow.
        if not is_variable:
            # Special handle for dictionaries
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            # Special handle for list of dictionaries
            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            # Special handle for strings
            elif expected_type_converted == str:
                # We don't check for case sensitivity for string, as long as it's not a variable
                result = string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        # Check if the value is within the possible answers
        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    # Check for optional parameters not provided but allowed
    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {repr(param)} not provided and not marked as optional."
            )
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_enforce_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: dict,
    language: str,
    model_name: str,
):
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_enforce_order:wrong_count",
        }

    func_name_list = list(possible_answers.keys())
    possible_answers_list = []

    for key, value in possible_answers.items():
        possible_answers_list.append({key: value})

    for i in range(len(possible_answers_list)):
        func_description = find_description(func_descriptions, func_name_list[i])

        result = simple_function_checker(
            func_description,
            model_output[i],
            possible_answers_list[i],
            language,
            model_name,
        )
        if not result["valid"]:
            return result

    return {"valid": True, "error": []}


def parallel_function_checker_no_order(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
    language: str,
    model_name: str,
):
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices = []

    # We go throught the possible answers one by one, and eliminate the model output that matches the possible answer
    # It must be this way because we need ground truth to fetch the correct function description
    for i in range(len(possible_answers)):
        # possible_answers[i] is a dictionary with only one key
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)

        all_errors = []

        for index in range(len(model_output)):
            if index in matched_indices:
                continue

            result = simple_function_checker(
                func_description,
                model_output[index],
                possible_answers[i],
                language,
                model_name,
            )

            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_output_item": model_output[index],
                            "possible_answer_item": possible_answers[i],
                        }
                    }
                )

        if not result["valid"]:
            considered_indices = [
                i for i in range(len(model_output)) if i not in matched_indices
            ]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} of model output for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def multiple_function_checker(
    func_descriptions: list,
    model_output: list,
    possible_answers: list,
    language: str,
    model_name: str,
):
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }

    # possible_answers is a list of only one dictionary with only one key
    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(
        func_description,
        model_output[0],
        possible_answers[0],
        language,
        model_name,
    )


#### Main function ####
def ast_checker(
    func_description,
    model_output,
    possible_answer,
    language,
    test_category: str,
    model_name: str,
):
    if "parallel" in test_category:
        return parallel_function_checker_no_order(
            func_description, model_output, possible_answer, language, model_name
        )

    elif "multiple" in test_category:
        return multiple_function_checker(
            func_description, model_output, possible_answer, language, model_name
        )

    else:
        if len(model_output) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }

        return simple_function_checker(
            func_description[0], model_output[0], possible_answer[0], language, model_name
        )


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def parse_json_function_call(source_code):
    json_match = re.search(r"\[\s*{.*?}\s*(?:,\s*{.*?}\s*)*\]", source_code, re.DOTALL)
    if json_match:
        source_code = json_match.group(0)

    try:
        json_dict = json.loads(source_code)
    except json.JSONDecodeError as e:
        return []

    function_calls = []
    for function_call in json_dict:
        if isinstance(function_call, dict):
            function_name = function_call["function"]
            arguments = function_call["parameters"]
            function_calls.append({function_name: arguments})
    return function_calls


def ast_parse(
    input_str: str,
    language: str,
    has_tool_call_tag: bool = False,
) -> list[dict]:
    if has_tool_call_tag:
        match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", input_str, re.DOTALL)
        if match:
            input_str = match.group(1).strip()
        else:
            raise ValueError(f"No tool call tag found in input string: {input_str}")

    if language == "python":
        # We only want to remove wrapping quotes that could have been added by the model.
        cleaned_input = input_str.strip().strip("'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted

    elif language == "json":
        json_match = re.search(r"\[.*\]", input_str, re.DOTALL)
        if json_match:
            input_str = json_match.group(0)
        return parse_json_function_call(input_str)

    else:
        raise NotImplementedError(f"Unsupported language: {language}")



def parse_fun_audio_chat_function_v1(text):
    tool_items = []
    tool_match_list = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    for tool_match in tool_match_list:
        tool_json = tool_match.strip()
        tool_json = tool_json.replace("\\n", " ")
        try:
            tool_item = json.loads(tool_json)
            tool_items.append(tool_item)
        except Exception as e:
            # print(f"tool_json={tool_json}")
            continue
    tool_items = [{i["name"]: i["arguments"]} for i in tool_items]
    return tool_items


def is_function_call_format_valid(decoded_output):
    # Ensure the output is a list of dictionaries
    if type(decoded_output) == list:
        for item in decoded_output:
            if type(item) != dict:
                return False
            # Check for `{func1: {param1: val1, param2: val2, ...}}`, should only have one key-value pair
            if len(item) != 1:
                return False
            # Check for `{param1: val1, param2: val2, ...}`; the parameter-value pairs should be a dictionary
            if type(list(item.values())[0]) != dict:
                return False
        return True
    return False


def output_record_table_v1(record_list, table_file):
    workbook = xlsxwriter.Workbook(table_file)
    worksheet = workbook.add_worksheet("SpeechFC")

    col_index = 0
    col_key = 1
    col_utterance = 2
    col_ground = 3
    col_predict = 4
    col_predict_raw = 5
    col_valid_flag = 6
    col_error_type = 7
    col_error_reason = 8

    title_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter'})
    wrap_format = workbook.add_format({"text_wrap": True, "align": "left", "valign": "vcenter"})
    english_letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                       "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    column_name_list = []
    column_name_list.extend(english_letters)
    for letter1 in english_letters:
        for letter2 in english_letters:
            column_name_list.append(f"{letter1}{letter2}")

    worksheet.write(0, col_index, '序号', title_format)
    worksheet.write(0, col_key, '任务编号', title_format)
    worksheet.write(0, col_utterance, '用户语句', title_format)
    worksheet.write(0, col_ground, '标注结果', title_format)
    worksheet.write(0, col_predict, '模型结果', title_format)
    worksheet.write(0, col_predict_raw, '模型结果(原始)', title_format)
    worksheet.write(0, col_valid_flag, '正确性', title_format)
    worksheet.write(0, col_error_type, '错误类型', title_format)
    worksheet.write(0, col_error_reason, '错误原因', title_format)

    worksheet.freeze_panes(1, 2)
    worksheet.set_column('A:A', 5, workbook.add_format({'align': 'center', 'valign': 'vcenter'}))
    worksheet.set_column('B:B', 15, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
    worksheet.set_column('C:C', 25, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
    worksheet.set_column('D:D', 30, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
    worksheet.set_column('E:E', 30, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
    worksheet.set_column('F:F', 30, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
    worksheet.set_column('G:G', 7, workbook.add_format({'align': 'center', 'valign': 'vcenter'}))
    worksheet.set_column('H:H', 25, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))
    worksheet.set_column('I:I', 25, workbook.add_format({'align': 'left', 'valign': 'vcenter'}))

    for record_idx, record_item in enumerate(record_list):
        raw_idx = record_idx + 1
        index = record_item["index"]
        utterance = record_item["utterance"]
        dataset_key = record_item["dataset_key"]

        valid_flag = record_item["valid_flag"]
        assert valid_flag in [True, False]
        valid_flag = 1 if valid_flag else 0

        error_type = record_item["error_type"]
        error_reasons = record_item["error_reasons"]
        error_reasons = "\n".join([f"{idx + 1}. {r}" for idx, r in enumerate(error_reasons)])
        ground_truth_result = record_item["ground_truth_result"]
        ground_truth_result = json.dumps(ground_truth_result, ensure_ascii=False)
        model_predict_result = record_item["model_predict_result"]
        model_predict_result = json.dumps(model_predict_result, ensure_ascii=False)
        model_predict_raw = record_item["model_predict_raw"]
        model_predict_raw = f"{model_predict_raw}"

        worksheet.write(raw_idx, col_index, index)
        worksheet.write(raw_idx, col_key, dataset_key, wrap_format)
        worksheet.write(raw_idx, col_utterance, utterance, wrap_format)
        worksheet.write(raw_idx, col_ground, ground_truth_result, wrap_format)
        worksheet.write(raw_idx, col_predict, model_predict_result, wrap_format)
        worksheet.write(raw_idx, col_predict_raw, model_predict_raw, wrap_format)
        worksheet.write(raw_idx, col_valid_flag, valid_flag)
        worksheet.write(raw_idx, col_error_type, error_type, wrap_format)
        worksheet.write(raw_idx, col_error_reason, error_reasons, wrap_format)

    workbook.close()


def tools2function_v1(tools):
    functions = []
    for tool_item in tools:
        function_name = tool_item["function"]["name"]
        function_description = tool_item["function"]["description"]
        function_parameters = tool_item["function"]["parameters"]
        function_item = {"name": function_name, "description": function_description, "parameters": function_parameters}
        functions.append(function_item)
    return functions


def compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder):
    result_file = os.path.join(base_folder, f"{model_name}/{dataset_name}/predict-{model_name}-{dataset_name}.json")
    record_file = os.path.join(base_folder, f"{model_name}/{dataset_name}/record-{model_name}-{dataset_name}.json")
    record_table = os.path.join(base_folder, f"{model_name}/{dataset_name}/record-{model_name}-{dataset_name}.xlsx")
    result_list = parse_json_line(result_file)

    correct_count = 0
    record_list = []
    error_type_counter = Counter()
    for result_item in tqdm(result_list):
        predict_result = result_item.get("predict", "")
        possible_answer_item, model_result_item = [], []
        possible_answer_item = result_item["ground_truth"]
        if isinstance(possible_answer_item, str):
            possible_answer_item = json.loads(possible_answer_item)
        prompt_function = result_item["function"]
        if isinstance(prompt_function, str):
            prompt_function = json.loads(prompt_function)
            prompt_function = tools2function_v1(prompt_function)

        try:
            model_result_item = parse_fun_audio_chat_function_v1(predict_result)

            for i in range(len(model_result_item)):
                model_result = model_result_item[i]
                update_model_result = {}
                for function_name, parameter_dict in model_result.items():
                    if isinstance(parameter_dict, list) and len(parameter_dict) == 1 and isinstance(parameter_dict[0], dict):
                        parameter_dict = parameter_dict[0]
                    assert isinstance(parameter_dict, dict)
                    update_model_result[function_name] = parameter_dict
                model_result_item[i] = update_model_result

        except:
            valid_flag = False
            error_type = "ast_decoder:decoder_failed"
            error_reasons = [f"Invalid syntax. Failed to decode AST. {str(predict_result)}"]
            result_item["valid_flag"] = valid_flag
            result_item["error_type"] = error_type
            result_item["error_reasons"] = error_reasons
            result_item["ground_truth_result"] = possible_answer_item
            result_item["model_predict_result"] = model_result_item
            result_item["model_predict_raw"] = predict_result
            record_list.append(result_item)
            error_type_counter.update([error_type])
            continue

        if not is_function_call_format_valid(model_result_item):
            valid_flag = False
            error_type = "ast_decoder:decoder_wrong_output_format"
            error_reasons = ["Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."]
            result_item["valid_flag"] = valid_flag
            result_item["error_type"] = error_type
            result_item["error_reasons"] = error_reasons
            result_item["ground_truth_result"] = possible_answer_item
            result_item["model_predict_result"] = model_result_item
            result_item["model_predict_raw"] = predict_result
            record_list.append(result_item)
            error_type_counter.update([error_type])
            continue

        checker_result = ast_checker(
            prompt_function,
            model_result_item,
            possible_answer_item,
            language="python",
            test_category=test_category,
            model_name=model_name,
        )

        if checker_result["valid"]:
            correct_count += 1
        else:
            pass

        valid_flag = checker_result["valid"]
        error_type = checker_result.get("error_type", "")
        error_type = "correct" if error_type == "" else error_type
        error_reasons = checker_result["error"]
        result_item["valid_flag"] = valid_flag
        result_item["error_type"] = error_type
        result_item["error_reasons"] = error_reasons
        result_item["ground_truth_result"] = possible_answer_item
        result_item["model_predict_result"] = model_result_item
        result_item["model_predict_raw"] = predict_result
        record_list.append(result_item)
        error_type_counter.update([error_type])

    save_json_line(record_list, record_file)
    output_record_table_v1(record_list, record_table)
    total_count = len(result_list)
    accuracy = correct_count / total_count if total_count != 0 else 0
    header = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
    }
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count} / {total_count})\n")

    # for error_type, frequency in error_type_counter.most_common():
    #     rate_error_type = frequency / total_count if total_count != 0 else 0
    #     print(f"{error_type}: {frequency} ({rate_error_type:.2%})")


def compute_metrics_example():
    base_folder = "../result/"
    model_name = "fun-audio-chat-s2t"

    dataset_name = "SpeechFC-BFCL-Single"
    test_category = "multiple"
    compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder)

    dataset_name = "SpeechFC-BFCL-Parallel"
    test_category = "parallel"
    compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder)

    dataset_name = "SpeechFC-SmartInteract"
    test_category = "multiple"
    compute_metrics_bfcl_v1(model_name, dataset_name, test_category, base_folder)


if __name__ == '__main__':
    compute_metrics_example()





