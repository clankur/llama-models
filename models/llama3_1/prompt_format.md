

# Llama 3.1 - Prompt Formats
## Tokens
Here is a list of special tokens that are supported by Llama 3.1:
- `<|begin_of_text|>`: Specifies the start of the prompt
- `<|end_of_text|>`: Model will cease to generate more tokens. This token is generated only by the base models.
- `<|finetune_right_pad_id|>`: This token is used for padding text sequences to the same length in a batch.
- `<|start_header_id|>` and `<|end_header_id|>`: These tokens enclose the role for a particular message. The possible roles are: [system, user, assistant and ipython]
- `<|eom_id|>`: End of message. A message represents a possible stopping point for execution where the model can inform the executor that a tool call needs to be made. This is used for multi-step interactions between the model and any available tools. This token is emitted by the model when the Environment: ipython instruction is used in the system prompt, or if the model calls for a built-in tool.
- `<|eot_id|>`: End of turn. Represents when the model has determined that it has finished interacting with the user message that initiated its response. This is used in two scenarios:
    - at the end of a direct interaction between the model and the user
    - at the end of multiple interactions between the model and any available tools
    This token signals to the executor that the model has finished generating a response.
- `<|python_tag|>`: Is a special tag used in the model's response to signify a tool call.



There are 4 different roles that are supported by Llama 3.1
- `system`: Sets the context in which to interact with the AI model. It typically includes rules, guidelines, or necessary information that helps the model respond effectively.
- `user`: Represents the human interacting with the model. It includes the inputs, commands, and questions to the model.
- `ipython`: A new role introduced in Llama 3.1. Semantically, this role means "tool". This role is used to mark messages with the output of a tool call when sent back to the model from the executor.
- `assistant`: Represents the response generated by the AI model based on the context provided in the `system`, `ipython` and `user` prompts.

## Llama 3.1 Base Model

Text completion for Llama 3.1 base model uses this format.

##### Input Prompt Format
```
<|begin_of_text|>Color of sky is blue but sometimes can also be
```

##### Model Response Format
```
 red, orange, yellow, green, purple, pink, brown, gray, black, white, and even rainbow colors. The color of the sky can change due to various reasons such as time of day, weather conditions, pollution, and atmospheric phenomena.
The color of the sky is primarily blue because of a phenomenon called
```



Note start special tag


## Llama 3.1 Instruct Model
## User and assistant conversation

Here is a regular multi-turn user assistant conversation and how its formatted.

##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Answer who are you in the form of jeopardy?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
Here's my response

"What is a helpful assistant?"<|eot_id|>
```






## Tool Calling Formats


The three built-in tools (brave_search, wolfram_alpha, and code interpreter) can be turned on using the system prompt:
- Brave Search: Tool call to perform web searches.
- Wolfram Alpha: Tool call to perform complex mathematical calculations.
- Code Interpreter: Enables the model to output python code.

## Builtin Tool Calling


Here is an example of a conversation using brave search


##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Tools: brave_search, wolfram_alpha
Cutting Knowledge Date: December 2023
Today Date: 21 September 2024

You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Search the web for the latest price of 1oz gold?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
<|python_tag|>brave_search.call(query="latest price of 1oz gold")<|eom_id|>
```




- Just including Environment: ipython turns on code interpreter; therefore, you don't need to specify code interpretation on the Tools: line. The model can generate python code which is interpreted by the executor, with the result provided back to the model.
- The message body of the assistant response starts with a special tag <|python_tag|>
- As alluded to above, in such an environment, the model can generate <|eom_id|> instead of just the standard <|eot_id|> . The latter indicates the turn is finished, while the former indicates continued multi-step reasoning. That is, the model is expecting a continuation message with the output of the tool call.
- The model tool call response is of the form `tool.call(query="...")` wher tool is `brave_search` or `wolfram_alpha`


## Builtin Code Interpreter

Here is an actual example of model responding with code

##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython<|eot_id|><|start_header_id|>user<|end_header_id|>

Write code to check if number is prime, use that to see if the number 7 is prime<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
<|python_tag|>def is_prime(n):
    if n <= 1
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

print(is_prime(7))  # Output: True<|eom_id|>
```




- Model starts with <|python_tag|> and continues writing python code that it needs to be executed
- No explicit mention of code_interpreter in system prompt. `Environment: ipython` implicitly enables it.


## Built-in tools full interaction

Here is a full interaction with the built-in tools including the tool response and the final assistant response.

##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Tools: brave_search, wolfram_alpha
<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the 100th decimal of pi?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|python_tag|>wolfram_alpha.call(query="100th decimal of pi")<|eom_id|><|start_header_id|>ipython<|end_header_id|>


{
    "queryresult": {
        "success": true,
        "inputstring": "100th decimal of pi",
        "pods": [
            {
                "title": "Input interpretation",
                "subpods": [
                    {
                        "title": "",
                        "plaintext": "100th digit | π"
                    }
                ]
            },
            {
                "title": "Nearby digits",
                "subpods": [
                    {
                        "title": "",
                        "plaintext": "...86208998628034825342117067982148086513282306647093..."
                    }
                ]
            },
            {
                "title": "Result",
                "primary": true,
                "subpods": [
                    {
                        "title": "",
                        "plaintext": "7"
                    }
                ]
            }
        ]
    }
}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
The 100th decimal of pi is 7.<|eot_id|>
```




- Note the `<|python_tag|>` in the assistant response.
- Role is `ipython` for the wolfram alpha response that is passed back to the model.
- Final message from assistant has <|eot_id|> tag.



## Zero shot tool calling
## JSON based tool calling


Llama models can now output custom tool calls from a single message to allow easier tool calling.
The following prompts provide an example of how custom tools can be called from the output of the model.
It's important to note that the model itself does not execute the calls; it provides structured output to facilitate calling by an executor.


##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython

Cutting Knowledge Date: December 2023
Today Date: 21 September 2024

You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Answer the user's question by making use of the following functions if needed.
If none of the function can be used, please say so.
Here is a list of functions in JSON format:
{
    "type": "function",
    "function": {
        "name": "trending_songs",
        "description": "Returns the trending songs on a Music site",
        "parameters": {
            "type": "object",
            "properties": [
                {
                    "n": {
                        "type": "object",
                        "description": "The number of songs to return"
                    }
                },
                {
                    "genre": {
                        "type": "object",
                        "description": "The genre of the songs to return"
                    }
                }
            ],
            "required": ["n"]
        }
    }
}

Return function calls in JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>

Use tools to get latest trending songs<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
<|python_tag|>{
    "type": "function",
    "name": "trending_songs",
    "parameters": {
        "n": "10",
        "genre": "all"
    }
}<|eom_id|>
```




- JSON format for providing tools needs name, description and parameters
- Model responds with `<|python_tag|>` and `<|eom_id|>` as `Environment: ipython` was in the system prompt
- Instructions for tools added as a user message
- Only single tool calls are supported as of now



## Example of a user defined tool calling
## `<function>` based tool calling


Here is an example of how you could also write custom instructions for model to do zero shot tool calling.
In this example, we define a custom tool calling format using the `<function>` tag.


##### Input Prompt Format
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython

Cutting Knowledge Date: December 2023
Today Date: 21 September 2024

You are a helpful assistant.
<|eot_id|><|start_header_id|>user<|end_header_id|>

You have access to the following functions:

Use the function 'trending_songs' to 'Returns the trending songs on a Music site':
{"name": "trending_songs", "description": "Returns the trending songs on a Music site", "parameters": {"genre": {"description": "The genre of the songs to return", "param_type": "str", "required": false}, "n": {"description": "The number of songs to return", "param_type": "int", "required": true}}}

Think very carefully before calling functions.
If you choose to call a function ONLY reply in the following format with no prefix or suffix:

<function=example_function_name>{"example_name": "example_value"}</function>

Reminder:
- If looking for real time information use relevant functions before falling back to brave_search
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line<|eot_id|><|start_header_id|>user<|end_header_id|>

Use tools to get latest trending songs<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

##### Model Response Format
```
<function=trending_songs>{"n": 10}</function><|eot_id|>
```




- In this case, model does NOT respond with `<|python_tag|>` and ends with `<|eot_id|>`
- Instructions for tools added as a user message


Thank You!