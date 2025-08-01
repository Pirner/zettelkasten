import random

import inspect
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_weather(location: str) -> str:
    """This tool returns the current weather situation.
    Args:

    Returns:
        str:"Weather situation"
        Example Response cloudy
        """
    # Specify stock ticker
    weather_situations = ['cloudy', 'rainy', 'sunny', 'foobar']
    return random.choice(weather_situations)


def add(x: float, y: float) -> float:
    """
    this function adds up the value of x and y
    :param x: first parameter of the addition
    :param y: second parameter of the addition
    :return: result of the addition of x and y
    """
    return x + y


def load_model(device='cuda'):
    model_id = r'C:\dev\llms\llama_3_2_1B_instruct'
    model_id = r'C:\dev\llms\llama_3_2_3B_instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    return model, tokenizer


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.
    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def select_tool(model, tokenizer, user_prompt):
    tools = [get_weather, add]
    tools_schema = [function_to_json(x) for x in tools]

    system_prompt = [
        {
            "role": "system",
            "content":
                f"""
                    You are an expert in composing functions. You are given a question and a set of possible functions. 
                    Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
                    If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
                    If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
                    You SHOULD NOT include any other text in the response.
                    Here is a list of functions in JSON format that you can invoke.
                    {tools_schema}
                    """
        },
    ]
    user_messages = [
        {"role": "user",
         "content": user_prompt},
    ]

    messages = system_prompt + user_messages

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        tools=tools_schema,
        format='JSON',
    ).to(model.device)
    tokens = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=float(0.2),
        top_k=1,
    )
    outputs = tokenizer.decode(tokens[0][inputs["input_ids"].shape[-1]:], skip_special_token=True)
    return outputs


def main():
    model, tokenizer = load_model()
    user_prompt = [
        {"role": "user",
         "content": "What is 2 + 5?"},
    ]
    outputs = select_tool(model, tokenizer, user_prompt)
    print(outputs)


if __name__ == '__main__':
    main()
