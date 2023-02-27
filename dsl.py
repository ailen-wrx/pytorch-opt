import ast, astunparse, re
import copy


class refactorHelper(object):
    def __init__(self, rules):
        self.rules = []
        for rule in rules:
            self.rules.append(Rule(rule))


class Rule(object):
    def __init__(self, rule):
        self.srcPatterns = []
        self.destPatterns = []
        self.arg_dict = {}
        self.detect_only = False

        src, dest = rule.split('=>')
        for src_assignment in src.split(';'):
            if src_assignment.strip() == "<any_assignment>":
                self.srcPatterns.append("<any_assignment>")
            else:
                self.srcPatterns.append(Assignment(src_assignment.strip(), self))
        if dest.strip() == "<do_not_refactor>":
            self.detect_only = True
            return
        for dest_assignment in dest.split(';'):
            self.destPatterns.append(Assignment(dest_assignment.strip(), self))

    def setArg(self, arg, val):
        self.arg_dict[arg] = val

    def patternMatch(self, call_statements):
        if len(call_statements) < len(self.srcPatterns):
            return -1, -1
        for i in range(len(call_statements) - len(self.srcPatterns) + 1):
            if self.subSequenceMatch(call_statements[i: i + len(self.srcPatterns)]):
                return i, i + len(self.srcPatterns)
        return -1, -1

    def subSequenceMatch(self, call_sequence):
        for idx in self.arg_dict:
            self.arg_dict[idx] = None

        for i in range(len(self.srcPatterns)):
            src_pattern = self.srcPatterns[i]
            if isinstance(src_pattern, str):
                continue
            call_stmt = call_sequence[i]
            if call_stmt.refactoredFunctionId is None:
                if src_pattern.functionCall.functionId.id != '.'.join(call_stmt.functionId):
                    return 0
            else:
                if not src_pattern.functionCall.functionId.accept_refactor or \
                        src_pattern.functionCall.functionId.id != '.'.join(call_stmt.refactoredFunctionId):
                    return 0

            if not src_pattern.output.assigned:
                idx = src_pattern.output.variable
                if idx not in self.arg_dict:
                    return 0
                elif not self.arg_dict[idx]:
                    self.arg_dict[idx] = call_stmt.output
                elif self.arg_dict[idx] != call_stmt.output:
                    return 0

            for arg_idx, arg in enumerate(src_pattern.functionCall.arguments):
                if not call_stmt.arguments or len(call_stmt.arguments) <= arg_idx:
                    return 0
                stmt_arg = call_stmt.arguments[arg_idx]
                if isinstance(arg, list):
                    if arg[0] == '+' and stmt_arg[0] == '#+':

                        if not arg[1].assigned:
                            idx = arg[1].variable
                            if idx not in self.arg_dict:
                                return 0
                            elif not self.arg_dict[idx]:
                                self.arg_dict[idx] = stmt_arg[1]
                            elif self.arg_dict[idx] != stmt_arg[1]:
                                return 0

                        if not arg[2].assigned:
                            idx = arg[2].variable
                            if idx not in self.arg_dict:
                                return 0
                            elif not self.arg_dict[idx]:
                                self.arg_dict[idx] = stmt_arg[2]
                            elif self.arg_dict[idx] != stmt_arg[2]:
                                return 0

                elif not arg.assigned:
                    idx = arg.variable
                    if idx not in self.arg_dict:
                        return 0
                    elif not self.arg_dict[idx]:
                        self.arg_dict[idx] = stmt_arg
                    elif self.arg_dict[idx] != stmt_arg:
                        return 0

            for kw in src_pattern.functionCall.keywords:
                if kw.key not in call_stmt.keywords:
                    return 0
                stmt_kw_arg = call_stmt.keywords[kw.key]
                if not kw.argument.assigned:
                    idx = kw.argument.variable
                    if idx not in self.arg_dict:
                        return 0
                    elif not self.arg_dict[idx]:
                        self.arg_dict[idx] = stmt_kw_arg
                    elif self.arg_dict[idx] != stmt_kw_arg:
                        return 0

        return 1

    def refactor(self, call_stmts):
        if self.detect_only:
            return None
        package_mngr = call_stmts[0].global_visitor.packageMngr
        trees = []
        for dest_pattern in self.destPatterns:
            tree = ast.parse('a=foo(x, y=z)').body[0]
            if dest_pattern.output.assigned:
                tree.targets[0].id = dest_pattern.output.value
            else:
                tree.targets[0].id = '.'.join(self.arg_dict[dest_pattern.output.variable])

            function_id = dest_pattern.functionCall.functionId.id
            if "self" not in function_id and function_id not in package_mngr.modules.values():
                import_tree = ast.parse('from a import b').body[0]
                import_tree.module = '.'.join(function_id.split('.')[:-1])
                import_tree.names[0].name = function_id.split('.')[-1]
                trees.append(import_tree)
                tree.value.func.id = import_tree.names[0].name

            argument = copy.deepcopy(tree.value.args[0])
            tree.value.args = []
            for arg in dest_pattern.functionCall.arguments:
                if arg.assigned:
                    tmp_argument = copy.deepcopy(argument)
                    tmp_argument.id = arg.value
                    tree.value.args.append(tmp_argument)
                else:
                    tmp_argument = copy.deepcopy(argument)
                    tmp_argument.id = '.'.join(self.arg_dict[arg.variable])
                    tree.value.args.append(tmp_argument)

            keyword = copy.deepcopy(tree.value.keywords[0])
            tree.value.keywords = []
            for kw in dest_pattern.functionCall.keywords:
                tmp_keyword = copy.deepcopy(keyword)
                tmp_keyword.arg = kw.key
                if kw.argument.assigned:
                    tmp_keyword.value.id = kw.argument.value
                    tree.value.keywords.append(tmp_keyword)
                else:
                    tmp_keyword.value.id = '.'.join(self.arg_dict[kw.argument.variable])
                    tree.value.keywords.append(tmp_keyword)
            trees.append(tree)

        return trees




class Assignment(object):
    def __init__(self, assignment, rule):
        self.output = ''
        self.functionCall = None
        self.Rule = rule
        functionCall, output = assignment.split('->')
        self.output = Argument(output.strip(), self.Rule)
        self.functionCall = FunctionCall(functionCall.strip(), self.Rule)


class FunctionCall(object):
    def __init__(self, func_call, rule):
        self.functionId = None
        self.arguments = []
        self.keywords = []
        self.Rule = rule

        parse = re.split(r'[(,)]', func_call)
        self.functionId = function_Id(parse[0])
        for arg in parse[1:-1]:
            if '=' in arg:
                self.keywords.append(Keyword(arg.strip(), self.Rule))
            elif '+' in arg:
                self.arguments.append(['+', Argument(arg.split('+')[0].strip(), self.Rule), Argument(arg.split('+')[1].strip(), self.Rule)])
            else:
                self.arguments.append(Argument(arg.strip(), self.Rule))


class Keyword(object):
    def __init__(self, argument, rule):
        self.key = ''
        self.argument = None
        self.Rule = rule

        key, arg = argument.split('=')
        self.key = key.strip()
        self.argument = Argument(arg.strip(), self.Rule)


class Argument(object):
    def __init__(self, argument, rule):
        self.assigned = False
        self.variable = None
        self.value = None
        self.Rule = rule

        self.assigned = not argument.startswith('$')
        if not self.assigned:
            self.variable = argument[1:]
            self.Rule.setArg(self.variable, None)
        else:
            self.value = argument


class function_Id(object):
    def __init__(self, func_id):
        if func_id.startswith('@'):
            self.id = func_id[1:]
            self.accept_refactor = True
        else:
            self.id = func_id
            self.accept_refactor = False


def main():
    pass


if __name__ == '__main__':
    main()
