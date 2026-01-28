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

import os
import re
import codecs
import json
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


def sum_key_list(data):
    key_counter = Counter()
    for dictionary in data:
        key_counter.update(dictionary.keys())
    key_count_dict = dict(key_counter)
    return key_count_dict


def find_description(func_descriptions, name):
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] in name:
                return func_description
        return None
    else:
        return func_descriptions


def get_possible_answer_type(possible_answer):
    if possible_answer != "":  # Optional parameter
        return type(possible_answer)
    return None


def standardize_string(input_string: str):

    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def flatten_dates(d):
    return {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in d.items()}


# 类别计算

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
    "list": list,
    "object": dict,
    "objectArray": list,
    "list": list,
    "list(string)": list,
    "list(enum)": list,
    "int": int,
    "enum": enumerate,
    "number": int
}

# This is the list of types that we need to recursively check its values
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple", "list(string)", "list(enum)", "object", "objectArray"]
NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]


def type_checker(
        param: str,
        value,
        possible_answer: list,
        expected_type_description: str,
        expected_type_converted,
        nested_type_converted,
        func_name,
):
    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error",
    }

    is_variable = False

    possible_answer_type = get_possible_answer_type(possible_answer)

    if possible_answer_type != None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    if value == "true":
        value = True
    if value == "false":
        value = False
    if type(value) == expected_type_converted:

        if nested_type_converted == None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param,
                            value_item,
                            possible_answer_item,
                            str(nested_type_converted),
                            nested_type_converted,
                            None,
                            func_name,
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
            result["error_type"] = "type_error"

    possible_answer_type = get_possible_answer_type(possible_answer)

    if possible_answer_type != None:

        if type(value) == possible_answer_type or possible_answer == value:
            result["is_variable"] = True
            return result

    output_value = type(value)
    result["valid"] = False
    result["error"] = [
        f"wrong type for parameter ({param}) of api ({func_name}):[excepted: {expected_type_converted}, real: {output_value}]"]
    result["error_type"] = "type_error"
    return result


def string_checker(param: str, model_output: str, possible_answer: list, function, question, test_category):
    func_name = function['name']
    standardize_model_output = standardize_string(model_output)
    standardize_possible_answer_item = standardize_string(possible_answer)

    if 'agent' in test_category:
        if standardize_model_output != standardize_possible_answer_item:
            return {
                "valid": False,
                "error": [
                    f"wrong value for parameter ({param}) of api ({func_name}): [excepted: {possible_answer}, real: [{model_output}]]"],
                "error_type": "value_error:string",
            }
    else:
        if (standardize_possible_answer_item not in standardize_model_output):
            return {
                "valid": False,
                "error": [
                    f"wrong value for parameter ({param}) of api ({func_name}): [excepted: {possible_answer}, real: [{model_output}]]"],
                "error_type": "value_error:string",
            }

    return {"valid": True, "error": []}


def list_checker(param: str, model_output: list, possible_answer: list, func_name):
    # Convert the tuple to a list

    standardize_model_output = list(model_output)

    # If the element in the list is a string, we need to standardize it
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = standardize_string(model_output[i])

    standardize_possible_answer = []

    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(
                standardize_string(possible_answer[i])
            )
        else:
            standardize_possible_answer.append(possible_answer[i])

    if standardize_model_output != standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"wrong value for parameter ({repr(param)}) of api ({func_name}): [expected {possible_answer}, real: [{repr(model_output)}]]"
            ],
            "error_type": "value_error:list/tuple",
        }

    return {"valid": True, "error": []}


def dict_checker(param: str, model_output: dict, possible_answers: list, func_name):
    # This function works for simple dictionaries, as well as dictionaries with nested dictionaries

    result = {"valid": True, "error": [], "error_type": ""}  # dict_checker:unclear

    possible_answer = possible_answers
    # possible_anwer is a single dictionary
    if not isinstance(model_output, dict):
        result["valid"] = False
        result["error"] = [
            f"wrong type for parameter ({param}) of api ({func_name}): [excepted: {possible_answer}, real: [{model_output}]]"]
        result["error_type"] = "value_error"
        return result
    else:
        if len(list(model_output.keys())) != len(list(possible_answer.keys())):
            result["valid"] = False
            result["error"] = [
                f"wrong value for parameter ({param}) of api ({func_name}): [excepted: {possible_answer}, real: [{model_output}]]"]
            result["error_type"] = "value_error"
            return result

        for key, value in model_output.items():
            if value == "true":
                value = True
            if value == "false":
                value = False
            if key not in possible_answer:
                result["valid"] = False
                result["error"] = [
                    f"wrong value for parameter ({param}) of api ({func_name}): [excepted: {possible_answer}, real: [{model_output}]]"]
                result["error_type"] = "value_error"
                return result

            expected_values = possible_answer[key]
            if isinstance(expected_values, dict):
                result = dict_checker(param, value, expected_values, func_name)
                if not result["valid"]:
                    return result
            else:
                standardize_value = value
                # If the value is a string, we need to standardize it
                if type(value) == str:
                    standardize_value = standardize_string(value)
                # We also need to standardize the possible answers
                standardize_possible_answer = []

                if type(possible_answer[key]) == str:
                    standardize_possible_answer.append(
                        standardize_string(possible_answer[key])
                    )
                else:
                    if type(possible_answer[key]) == dict:
                        standardize_possible_answer.append(flatten_dates(possible_answer[key]))
                    else:
                        standardize_possible_answer.append(possible_answer[key])

                if isinstance(standardize_possible_answer, list):
                    standardize_possible_answer = standardize_possible_answer[0]
                if str(standardize_possible_answer) not in str(standardize_value):
                    result["valid"] = False
                    result["error"] = [
                        f"wrong value for parameter ({param}) of api ({func_name}): [excepted: {possible_answer}, real: [{model_output}]]"]
                    result["error_type"] = "value_error"
                    return result

    return result


def list_dict_checker(param: str, model_output: list, possible_answers: list, func_name):
    result = {"valid": True, "error": [], "error_type": ""}  # list_dict_checker:unclear

    if len(model_output) != len(possible_answers):
        result["valid"] = False
        result["error"] = [
            f"wrong value for parameter ({param}) of api ({func_name}): [excepted: {possible_answers}, real: [{model_output}]]"]
        result["error_type"] = "value_error:list_dict_count"
        return result

    for dict_index in range(len(model_output)):
        if dict_index >= len(possible_answers):
            break
        result = dict_checker(
            param,
            model_output[dict_index],
            possible_answers[dict_index],
            func_name
        )
        if not result["valid"]:
            return result

    return result


# 场景计算
def simple_function_checker(
        func_description: dict,
        model_output: dict,
        possible_answers: dict,
        question: str,
        test_category: str
):
    # Extract function name and parameters details
    result = {
        "valid": True,
        "error": [],
        "error_type": "",
    }

    # When the function's reference parameter is empty, such as APIname()
    possible_answer = list(possible_answers.values())[0]

    if list(model_output.values())[0] == {} and func_description["parameters"] == {}:
        return result
    elif list(model_output.values())[0] == {} or func_description["parameters"] == {}:
        result["valid"] = False
        result["error_type"] = "wrong_param"
        return result

    if possible_answer == func_description["parameters"]['properties']:
        return result
    elif possible_answer == {} or func_description["parameters"] == {}:
        result["valid"] = False
        result["error_type"] = "wrong_param"
        return result

    # Function name error
    func_name = func_description["name"]
    if func_name not in model_output:
        result["valid"] = False
        result["error"] = [{"wrong_function": {"expected": func_name, "real": list(model_output.keys())[0]}}]
        result["error_type"] = "wrong_function_name"

        return result

    model_params = model_output[func_name]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]

    # Save the status of each check for later calculation

    for param in required_params:

        if param not in model_params:
            result = {"valid": False, "error": f"lack required_params: {param}", "error_type": "lack_args"}
            return result

    for param, value in model_params.items():
        # One extra parameter, add one 0
        if param not in param_details or param not in possible_answer:
            result = {"valid": False, "error": f"addition params: {param}", "error_type": "addition_args"}
            return result

        full_param_details = param_details[param]
        # Parameter type when the function is defined
        expected_type_description = full_param_details["type"]  # This is a string
        is_variable = False
        nested_type_converted = None

        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        # Handle special data types?
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            try:
                nested_type = param_details[param]["items"]["type"]
            except Exception as e:
                if "string" in param_details[param]["type"]:
                    nested_type = 'string'
                elif "enum" in param_details[param]["type"]:
                    nested_type = 'enum'
                else:
                    nested_type = 'dict'
            nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)

        # Allow python auto conversion from int to float
        if (
                expected_type_description == "float"
                and type(value) == int
        ):
            value = float(value)

        type_check_result = type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
            func_name,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            result = {"valid": False, "error": type_check_result["error"],
                      "error_type": type_check_result["error_type"]}
            return result

        if not is_variable:
            # Special handle for dictionaries
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param], func_name)
                if not result["valid"]:
                    result = {"valid": False, "error": result["error"], "error_type": result["error_type"]}
                    return result


            # Special category object_array
            elif expected_type_converted == list and nested_type_converted == dict:
                if expected_type_description == 'objectArray':
                    if len(value) != len(possible_answer[param]):
                        result = {"valid": False, "error": ["Wrong number of parameters for dictionary."],
                                  "error_type": "value_error:dict_items"}
                        return result

                    if not (all(dict_checker(param, val, pos)[0]["valid"] == True for val, pos in
                                zip(value, possible_answer[param]))):
                        result = {"valid": False, "error": ["Something wrong with specific item"],
                                  "error_type": "value_error:dict_items"}
                        return result

                result = list_dict_checker(param, value, possible_answer[param], func_name)
                if not result["valid"]:
                    result = {"valid": False, "error": result["error"], "error_type": result["error_type"]}
                    return result

            # Special handle for strings
            elif expected_type_converted == str:
                # We don't check for case sensitivity for string, as long as it's not a variable
                result = string_checker(param, value, possible_answer[param], func_description, question, test_category)
                if not result["valid"]:
                    result = {"valid": False, "error": result["error"], "error_type": result["error_type"]}
                    return result


            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param], func_name)
                if not result["valid"]:
                    result = {"valid": False, "error": result["error"], "error_type": result["error_type"]}
                    return result
    return result


def normal_checker(
        func_descriptions: list,
        model_output: list,
        possible_answers: dict,
        question: str,
        test_category: str,
):
    result = {}
    result["valid"] = True

    result_list = []
    if len(model_output) != len(possible_answers):
        result = {
            "valid": False,
            "error": ["The number of functions does not match the answer."],
            "error_type": "wrong functions number",
        }
        result_list.append(result)
        return result

    func_name_list = list(possible_answers.keys())
    possible_answers_list = []

    for key, value in possible_answers.items():
        possible_answers_list.append({key: value})

    if len(possible_answers_list) > 1:
        for index in range(len(possible_answers_list)):
            current_dict = possible_answers_list[index]
            keys_to_update = list(current_dict.keys())  # Get all keys
            for key in keys_to_update:
                new_key = re.sub(r'_\d+$', '', key)
                # If the key has changed, update the key and retain the value
                if new_key != key:
                    current_dict[new_key] = current_dict.pop(key)  # Move the old key-value to the new key

    output_list = sum_key_list(model_output)
    answer_list = sum_key_list(possible_answers_list)

    for name, count in output_list.items():
        if name not in answer_list:
            result = {
                "valid": False,
                "error": [f"extra function detected: {name} is not in the ground truth"],
                "error_type": "function_mismatch",
            }
            return result

    for name, count in answer_list.items():
        if name not in output_list:
            result = {
                "valid": False,
                "error": [f"extra function detected: {name} is not in the ground truth"],
                "error_type": "function_mismatch",
            }
            return result

    for name, count in output_list.items():
        if name not in answer_list or count != answer_list[name]:
            number = answer_list[name] if name in answer_list else 0
            result = {
                "valid": False,
                "error": [f"incorrect count for function {name}: [expected: {number}, actual: {count}]"],
                "error_type": "function_mismatch",
            }
            return result

    for i in range(len(possible_answers_list)):
        func_description = find_description(func_descriptions, func_name_list[i])
        for j in range(len(model_output)):
            if list(model_output[j].keys())[0] == list(possible_answers_list[i].keys())[0]:
                result = simple_function_checker(
                    func_description,
                    model_output[j],
                    possible_answers_list[i],
                    question,
                    test_category
                )
                if result["valid"]:
                    break
            else:
                result = {
                    "valid": False,
                    "error": ["wrong_function"],
                    "error_type": "simple_function_checker:unclear",
                }

        if not result["valid"]:
            return result

    return result


def parse_fun_audio_chat_function_v1(text):
    tool_items = []
    tool_match_list = re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    for tool_match in tool_match_list:
        tool_json = tool_match.strip()
        tool_json = tool_json.replace("\\n", " ")
        try:
            tool_item = json.loads(tool_json)
        except Exception as e:
            # print(f"tool_json={repr(tool_json)}")
            continue
        tool_items.append(tool_item)
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


def compute_metrics_acebench_v1(model_name, dataset_name, test_category, base_folder):
    result_file = os.path.join(base_folder, f"{model_name}/{dataset_name}/predict-{model_name}-{dataset_name}.json")
    record_file = os.path.join(base_folder, f"{model_name}/{dataset_name}/record-{model_name}-{dataset_name}.json")
    record_table = os.path.join(base_folder, f"{model_name}/{dataset_name}/record-{model_name}-{dataset_name}.xlsx")
    result_list = parse_json_line(result_file)

    correct_count = 0
    record_list = []
    error_type_counter = Counter()
    for result_item in tqdm(result_list):
        question = result_item["utterance"]
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

        if type(possible_answer_item) != list:
            possible_answer_item = [possible_answer_item]

        correct_flag = False
        all_errors, checker_result = [], {}
        # Filter from multiple candidate answers
        for possible_answer_item_ in possible_answer_item:

            checker_result = normal_checker(
                prompt_function,
                model_result_item,
                possible_answer_item_,
                question,
                test_category,
            )

            if checker_result["valid"]:
                correct_flag = True
                correct_count += 1
                break
            else:
                all_errors.append({
                    "error": checker_result["error"],
                    "error_type": checker_result["error_type"],
                })

        if not correct_flag and all_errors:
            checker_result = {
                "id": id,
                "valid": False,
                "error": all_errors[0]["error"],
                "error_type": all_errors[0]["error_type"],
                # "model_result": model_result_item_raw,
                # "possible_answer": possible_answer_item_,
            }

        valid_flag = checker_result["valid"]
        error_type = checker_result.get("error_type", "")
        error_type = "correct" if error_type == "" else error_type
        error_reasons = checker_result["error"]
        if not isinstance(error_reasons, list):
            error_reasons = [error_reasons]
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
    accuracy = round((correct_count / total_count if total_count != 0 else 0), 3)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count} / {total_count})\n")

    # for error_type, frequency in error_type_counter.most_common():
    #     rate_error_type = frequency / total_count if total_count != 0 else 0
    #     print(f"{error_type}: {frequency} ({rate_error_type:.2%})")


def compute_metrics_acebench_example():
    base_folder = "../result/"
    model_name = "fun-audio-chat-s2t"

    dataset_name = "SpeechFC-ACEBench-Single"
    test_category = "multiple"
    compute_metrics_acebench_v1(model_name, dataset_name, test_category, base_folder)

    dataset_name = "SpeechFC-ACEBench-Parallel"
    test_category = "multiple"
    compute_metrics_acebench_v1(model_name, dataset_name, test_category, base_folder)


if __name__ == '__main__':
    compute_metrics_acebench_example()











