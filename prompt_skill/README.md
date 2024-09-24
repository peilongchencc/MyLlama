# string.Template

项目中llm程序少不了要构建prompt，如何更方便的构建prompt可以帮助我们提升工作效率、减少错误拼接。本章就介绍 Python 标准库 `string` 模块中 `string.Template` 的使用。
- [string.Template](#stringtemplate)
  - [简介:](#简介)
  - [基础用法](#基础用法)
      - [基本示例](#基本示例)
      - [基本示例: 如何在大模型的Prompt中使用](#基本示例-如何在大模型的prompt中使用)
  - [substitute(替换) 和 safe\_substitute(安全替换) 的区别](#substitute替换-和-safe_substitute安全替换-的区别)
      - [示例: 使用 `substitute`](#示例-使用-substitute)
      - [示例: 使用 `safe_substitute`](#示例-使用-safe_substitute)
  - [动态构建复杂字符串](#动态构建复杂字符串)
  - [批量生成文本](#批量生成文本)
  - [结合字典构建多种prompt:](#结合字典构建多种prompt)
  - [与其他字符串格式化方式对比:](#与其他字符串格式化方式对比)


## 简介:

`string.Template` 提供了一种简单而强大的字符串模板功能，特别适合在需要动态生成文本的场景中使用，它主要通过使用占位符来实现对字符串的动态替换。

## 基础用法

`string.Template` 的占位符以 `$` 开头，后跟变量名，支持简单的字符串替换。

#### 基本示例

```python
from string import Template

# 创建一个模板字符串
template = Template("Hello, $name! You are welcome to $place.")

# 使用 substitute 进行字符串替换
result = template.substitute(name="Alice", place="Wonderland")

print(result)
```

终端输出: 

```log
Hello, Alice! You are welcome to Wonderland.
```

#### 基本示例: 如何在大模型的Prompt中使用

假设你需要为一个聊天机器人动态生成 Prompt，来回答关于数据库查询的问题，可以使用类似下面的模板: 

```python
from string import Template

# 定义大模型的 Prompt 模板
prompt_template = Template("""
You are a helpful assistant. The user wants to perform a query on the $database database.
Here are the user's instructions:
$instructions
""")

# 定义实际的变量值
variables = {
    "database": "MySQL",
    "instructions": "Select all users who joined after 2020."
}

# 使用 substitute 方法生成最终的 Prompt
prompt = prompt_template.substitute(variables)

print(prompt)
```

输出结果: 

```log
You are a helpful assistant. The user wants to perform a query on the MySQL database.
Here are the user's instructions:
Select all users who joined after 2020.
```


## substitute(替换) 和 safe_substitute(安全替换) 的区别

- `substitute`: 如果模板中存在某些未提供值的占位符，会抛出 `KeyError`。

- `safe_substitute`: 不会抛出错误，而是保留未提供值的占位符原样输出。

#### 示例: 使用 `substitute`

```python
from string import Template

template = Template("Hello, $name! Welcome to $place.")
# 仅提供 name 变量
result = template.substitute(name="Alice")
print(result)
```

终端输出:

```log
KeyError: 'place'
```

#### 示例: 使用 `safe_substitute`

```python
from string import Template

template = Template("Hello, $name! Welcome to $place.")
# 仅提供 name 变量
result = template.safe_substitute(name="Alice")
print(result)
```

终端输出: 

```log
Hello, Alice! Welcome to $place.
```


## 动态构建复杂字符串

通过`string.Template`，你可以轻松地构建具有多个部分的复杂字符串，例如: 

```python
from string import Template

email_template = Template("""
Dear $name,

Thank you for your interest in our $product.
We are happy to offer you a special discount: $discount.

Best regards,
The $company Team
""")

data = {
    "name": "Alice",
    "product": "Smartphone",
    "discount": "10% off",
    "company": "TechCorp"
}

email_content = email_template.substitute(data)
print(email_content)
```

终端输出: 

```log
Dear Alice,

Thank you for your interest in our Smartphone.
We are happy to offer you a special discount: 10% off.

Best regards,
The TechCorp Team
```


## 批量生成文本

如果你有多个不同的数据集，可以通过 `Template` 批量生成文本，这非常适合需要频繁生成动态内容的场景。

```python
from string import Template

template = Template("Hello, $name! Welcome to $place.")

users = [
    {"name": "Alice", "place": "Wonderland"},
    {"name": "Bob", "place": "Atlantis"},
    {"name": "Charlie", "place": "Neverland"}
]

for user in users:
    print(template.substitute(user))
```

终端输出: 

```log
Hello, Alice! Welcome to Wonderland.
Hello, Bob! Welcome to Atlantis.
Hello, Charlie! Welcome to Neverland.
```


## 结合字典构建多种prompt:

```python
from string import Template

my_dict = {
    # python字符串的隐式拼接技巧
    "single_choice_questions_type":
        '单选题：\n'
        '问题：$current_question\n'
        '选项：$current_answer\n'
        '病人回复：$asr_answer\n'
    ,
    # python括号形式拼接多行字符串
    "multiple_choice_questions_type":
        ('多选题：\n'
        '问题：$current_question\n'
        '选项：$current_answer\n'
        '病人回复：$asr_answer\n')
}

# 示例数据
single_choice_data = {
    "current_question": "你今天感觉如何？",
    "current_answer": "1. 好 2. 一般 3. 不好",
    "asr_answer": "1"
}

multiple_choice_data = {
    "current_question": "你有以下哪些症状呢？",
    "current_answer": "1. 头疼 2. 发烧 3. 四肢无力",
    "asr_answer": "12"
}

# 定义一个函数，根据类型选择模板并生成结果
def generate_question_text(question_type, data):
    # 获取对应的字符串模板
    question_template_str = my_dict.get(question_type)
    
    if question_template_str:
        # 使用 Template 构建模板
        question_template = Template(question_template_str)
        # 返回替换后的字符串
        return question_template.substitute(data)
    else:
        return "无效的题目类型"

# 根据实际情况选择题目类型，例如单选题
question_type = "single_choice_questions_type"
result = generate_question_text(question_type, single_choice_data)
print(result)

# 多选题
question_type = "multiple_choice_questions_type"
result = generate_question_text(question_type, multiple_choice_data)
print(result)
```

终端输出:

```log
单选题：
问题：你今天感觉如何？
选项：1. 好 2. 一般 3. 不好
病人回复：1

多选题：
问题：你有以下哪些症状呢？
选项：1. 头疼 2. 发烧 3. 四肢无力
病人回复：12
```


## 与其他字符串格式化方式对比:

与 `f-string` 或 `str.format()` 相比，`string.Template` 提供了更加简单的语法和灵活的扩展方式，特别适合那些需要在不同上下文中动态生成大量文本的场景。

```markdown
| 格式化方法         | 优点                                                                      | 缺点                                   |
|------------------|---------------------------------------------------------------------------|--------------------------------------|
| **f-string**     | 适合简单的、即席的字符串格式化，性能较好                                   | 不适合模板化的场景                    |
| **str.format()** | 功能强大                                                                   | 语法较为复杂，处理模板时较为冗长        |
| **string.Template** | 简单易用，适合需要频繁重复生成的场景，并且具备更好的扩展性                   |                                    |
```