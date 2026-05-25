
def Nargout(*args):
    """
    Эмуляция MATLAB-функции nargout для Python.
    Возвращает только те выходные аргументы, которые запрошены пользователем.

    Принцип работы:
    1. Анализирует стек вызовов, чтобы найти строку, где была вызвана эта функция
    2. Разбирает левую часть присваивания: "a, b, c = Nargout(...)"
    3. Считает количество переменных через запятую
    4. Возвращает соответствующее количество первых аргументов
    """
    import traceback
    callInfo = traceback.extract_stack()
    callLine = str(callInfo[-3].line)
    split_equal = callLine.split("=")
    split_comma = split_equal[0].split(",")
    num = len(split_comma)
    return args[0:num] if num > 1 else args[0]