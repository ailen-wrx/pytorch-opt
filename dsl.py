import ast, astunparse, re


class refactor_DSL(object):
    def __init__(self, rules):
        self.rules = []
        for rule in rules:
            self.rules.append(DSL_Rule(rule))


class DSL_Rule(object):
    def __init__(self, rule):
        self.srcPatterns = []
        self.destPatterns = []
        self.arg_dict = {}

        src, dest = rule.split('=>')
        for src_assignment in src.split(';'):
            self.srcPatterns.append(DSL_Assignment(src_assignment.strip(), self))
        for dest_assignment in dest.split(';'):
            self.destPatterns.append(DSL_Assignment(dest_assignment.strip(), self))

    def setArg(self, arg, val):
        self.arg_dict[arg] = val

    def patternMatch(self, call_statements):
        if len(call_statements) < len(self.srcPatterns):
            return None
        for i in range(len(call_statements) - len(self.srcPatterns)):
            for j in range(len(self.srcPatterns)):
                self.subSequenceMatch(call_statements[i: i + len(self.srcPatterns)], self.srcPatterns)

    def subSequenceMatch(self, call_sequence, pattern):
        for i in range(len(pattern)):
            if call_sequence[i].unimplemented: return None
            if pattern[i].functionCall.functionId.id != \
               '.'.join(call_sequence[i].functionId):
                return None
        return 1



class DSL_Assignment(object):
    def __init__(self, assignment, rule):
        self.output = ''
        self.functionCall = None
        self.Rule = rule
        functionCall, output = assignment.split('->')
        self.output = DSL_Argument(output.strip(), self.Rule)
        self.functionCall = DSL_FunctionCall(functionCall.strip(), self.Rule)


class DSL_FunctionCall(object):
    def __init__(self, func_call, rule):
        self.functionId = None
        self.arguments = []
        self.keywords = []
        self.Rule = rule

        parse = re.split(r'[(,)]', func_call)
        self.functionId = function_Id(parse[0])
        for arg in parse[1:-1]:
            if '=' in arg:
                self.keywords.append(DSL_Keyword(arg.strip(), self.Rule))
            else:
                self.arguments.append(DSL_Argument(arg.strip(), self.Rule))


class DSL_Keyword(object):
    def __init__(self, argument, rule):
        self.key = ''
        self.argument = None
        self.Rule = rule

        key, arg = argument.split('=')
        self.key = key.strip()
        self.argument = DSL_Argument(arg.strip(), self.Rule)


class DSL_Argument(object):
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
        self.id = func_id
        self.fullQual = ''


def main():
    pass


if __name__ == '__main__':
    main()
