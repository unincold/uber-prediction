import os
import re
import ast
import base64

def is_image_path(text):
    # 检查输入文本是否以典型的图像文件扩展名结尾
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    if text.endswith(image_extensions):
        return True
    else:
        return False

def encode_image(image_path):
    """将图像文件编码为base64。"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def is_url_or_filepath(input_string):
    # 检查input_string是否为URL
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    if url_pattern.match(input_string):
        return "URL"

    # 检查input_string是否为文件路径
    file_path = os.path.abspath(input_string)
    if os.path.exists(file_path):
        return "File path"

    return "Invalid"

def extract_data(input_string, data_type):
    # 正则表达式提取从'```python'开始到结束的内容，如果没有关闭的反引号
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    # 提取内容
    # re.DOTALL允许'.'匹配换行符
    matches = re.findall(pattern, input_string, re.DOTALL)
    # 返回第一个匹配项（如果存在），修剪空白并忽略潜在的关闭反引号
    return matches[0][0].strip() if matches else input_string

def parse_input(code):
    """使用AST解析输入字符串并提取函数名、参数和关键字参数。"""

    def get_target_names(target):
        """递归获取赋值目标中的所有变量名。"""
        if isinstance(target, ast.Name):
            return [target.id]
        elif isinstance(target, ast.Tuple):
            names = []
            for elt in target.elts:
                names.extend(get_target_names(elt))
            return names
        return []

    def extract_value(node):
        """提取AST节点的实际值"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            # TODO: 更好的处理变量的方法
            raise ValueError(
                f"Arguments should be a Constant, got a variable {node.id} instead."
            )
        # 添加其他需要处理的AST节点类型
        return None

    try:
        # 解析输入的代码字符串，生成AST（抽象语法树）
        tree = ast.parse(code)
        # 遍历AST中的所有节点
        for node in ast.walk(tree):
            # 如果节点是赋值语句
            if isinstance(node, ast.Assign):
                targets = []
                # 获取赋值语句的目标变量名
                for t in node.targets:
                    targets.extend(get_target_names(t))
                # 如果赋值语句的值是函数调用
                if isinstance(node.value, ast.Call):
                    # 获取函数名
                    func_name = node.value.func.id
                    # 获取函数调用的参数
                    args = [ast.dump(arg) for arg in node.value.args]
                    # 获取函数调用的关键字参数
                    kwargs = {
                        kw.arg: extract_value(kw.value) for kw in node.value.keywords
                    }
                    # 打印解析结果
                    print(f"Input: {code.strip()}")
                    print(f"Output Variables: {targets}")
                    print(f"Function Name: {func_name}")
                    print(f"Arguments: {args}")
                    print(f"Keyword Arguments: {kwargs}")
            # 如果节点是表达式且值是函数调用
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                targets = []
                # 获取函数名
                func_name = extract_value(node.value.func)
                # 获取函数调用的参数
                args = [extract_value(arg) for arg in node.value.args]
                # 获取函数调用的关键字参数
                kwargs = {kw.arg: extract_value(kw.value) for kw in node.value.keywords}

    except SyntaxError:
        print(f"Input: {code.strip()}")
        print("No match found")

    return targets, func_name, args, kwargs

if __name__ == "__main__":
    import json
    # 示例JSON字符串
    s='{"Thinking": "The Docker icon has been successfully clicked, and the Docker application should now be opening. No further actions are required.", "Next Action": None}'
    # 解析JSON字符串
    json_str = json.loads(s)
    print(json_str)