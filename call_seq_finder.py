import ast, gast, astunparse, importlib, inspect
import copy

from python_graphs import control_flow
from package_mngr import packageMngr


FLG_BINOP_ADD = '#+'
FLG_TUPLE     = '#T'


def unparse(node):
    source = astunparse.unparse(node)
    return (
        source.strip()
        .rstrip(' \n')
        .lstrip(' \n')
    )


def unparse_attribute(attribute):
    if isinstance(attribute, ast.Attribute):
        res = [attribute.attr]
        while isinstance(attribute.value, ast.Attribute):
            res = [attribute.value.attr] + res
            attribute = attribute.value
        if isinstance(attribute.value, ast.Attribute) or \
                isinstance(attribute.value, ast.Name):
            res = [attribute.value.id] + res
            return res
        elif isinstance(attribute.value, str):
            res = [attribute.value] + res
            return res
        else:
            res = [unparse(attribute.value)] + res
            return res
    elif isinstance(attribute, ast.Name):
        return [attribute.id]
    elif isinstance(attribute, ast.Tuple):
        attr = [FLG_TUPLE]
        for elt in attribute.elts:
            attr += unparse_attribute(elt)
        return attr
    elif isinstance(attribute, str):
        return [attribute]
    elif isinstance(attribute, ast.BinOp):
        if isinstance(attribute.op, ast.Add):
            return [FLG_BINOP_ADD, unparse_attribute(attribute.left), unparse_attribute(attribute.right)]

    return [unparse(attribute)]


def parse_attribute(attr):
    if attr[0] == FLG_BINOP_ADD:
        return f'{parse_attribute(attr[1])} + {parse_attribute((attr[2]))}'
    elif attr[0] == FLG_TUPLE:
        pass
    else:
        return '.'.join(attr)


class callStatement(object):
    def __init__(self, node, module_visitor):
        self.global_visitor = module_visitor
        self.ast_node = node
        self.output = None
        self.functionId = None
        self.refactoredFunctionId = None
        self.arguments = None
        self.keywords = None

    def set_output(self, output):
        # if isinstance(output, ast.Tuple):
        #     pass
        self.output = unparse_attribute(output)

    def set_argument(self, arg):
        if not self.arguments:
            self.arguments = []
        self.arguments.append(arg)

    def set_keyword(self, kw):
        if not self.keywords:
            self.keywords = {}
        self.keywords[kw.arg] = unparse_attribute(kw.value)

    def set_function(self, func):
        func_attributes = unparse_attribute(func)
        full_qual_func_id = self.global_visitor.packageMngr.lookUp(func_attributes[0])
        if full_qual_func_id is not None:
            func_attributes = full_qual_func_id.split('.') + func_attributes[1:]
        self.functionId = func_attributes

        # Refactoring
        if len(self.global_visitor.class_def_stack) == 0:
            if self.global_visitor.current_func.name in self.global_visitor.global_func_var_def and \
                    '.'.join(func_attributes) in self.global_visitor.global_func_var_def[self.global_visitor.current_func.name]:
                self.refactoredFunctionId = self.global_visitor.global_func_var_def[self.global_visitor.current_func.name]['.'.join(func_attributes)]
        else:
            class_info = self.global_visitor.class_def_stack[-1]
            if func_attributes[0] == 'self':
                self.refactoredFunctionId = class_info.look_up_attribute('.'.join(func_attributes[1:]))
            else:
                self.refactoredFunctionId = class_info.look_up_variable(self.global_visitor.current_func.name, '.'.join(func_attributes))

        # Save variables
        if self.output is not None:
            if len(self.global_visitor.class_def_stack) == 0:
                if self.global_visitor.current_func.name not in self.global_visitor.global_func_var_def:
                    self.global_visitor.global_func_var_def[self.global_visitor.current_func.name] = {}
                self.global_visitor.global_func_var_def[self.global_visitor.current_func.name][
                    '.'.join(self.output)] = self.functionId
            else:
                class_info = self.global_visitor.class_def_stack[-1]
                if self.output[0] == 'self':
                    class_info.set_attribute('.'.join(self.output[1:]), self.functionId)
                else:
                    class_info.set_variable(self.global_visitor.current_func.name, '.'.join(self.output),
                                            self.functionId)


class callVisitor(ast.NodeVisitor):
    def __init__(self, node, module_visitor):
        self.global_visitor = module_visitor
        self.root = node
        self.temp_call_stmt = callStatement(node, module_visitor)

    def debug(self, node):
        print(unparse(node))

    def dump_call_stmt(self):
        return self.temp_call_stmt

    def visit_Import(self, node):
        self.global_visitor.packageMngr.register_tmp(node)

    def visit_ImportFrom(self, node):
        self.global_visitor.packageMngr.register_tmp(node)

    def visit_Call(self, node):
        # self.debug(node)
        for arg in node.args:
            if isinstance(arg, ast.Call):
                call_visitor = callVisitor(arg, self.global_visitor)
                tmp_variable = self.global_visitor.get_temp_variable()
                call_visitor.temp_call_stmt.set_output(tmp_variable)
                call_visitor.visit(arg)
                tree = ast.parse(f'{tmp_variable} = foo(x)')
                arg = copy.deepcopy(tree.body[0].targets[0])
            self.temp_call_stmt.set_argument(unparse_attribute(arg))
        for kw in node.keywords:
            if isinstance(kw.value, ast.Call):
                call_visitor = callVisitor(kw.value, self.global_visitor)
                tmp_variable = self.global_visitor.get_temp_variable()
                call_visitor.temp_call_stmt.set_output(tmp_variable)
                call_visitor.visit(kw.value)
                tree = ast.parse(f'{tmp_variable} = foo(x)')
                kw.value = copy.deepcopy(tree.body[0].targets[0])
            self.temp_call_stmt.set_keyword(kw)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
            tmp_variable = self.global_visitor.get_temp_variable()
            new_node = ast.parse(f'{tmp_variable} = {astunparse.unparse(node.func.value).strip()}').body[0]
            call_visitor = callVisitor(new_node, self.global_visitor)
            call_visitor.temp_call_stmt.set_output(tmp_variable)
            call_visitor.visit(node.func.value)
            tree = ast.parse(f'{tmp_variable} = foo(x)')
            node.func.value = copy.deepcopy(tree.body[0].targets[0])
        self.temp_call_stmt.set_function(node.func)
        self.global_visitor.tmp_call_sequence.append_stmt(self)

    def visit_Assign(self, node):
        # self.debug(node)
        self.temp_call_stmt.set_output(node.targets[0])
        self.visit(node.value)


class callSequence(object):
    def __init__(self, module_visitor):
        self.global_visitor = module_visitor
        self.call_statements = []
        self.function_def = module_visitor.current_func
        self.class_def = copy.deepcopy(module_visitor.class_def_stack)

    def append_stmt(self, call_visitor):
        stmt = call_visitor.dump_call_stmt()
        self.call_statements.append(stmt)

    def print(self, i=0, j=-1):
        if j == -1:
            j = len(self.call_statements)
        # print(f'{self.function_def.lineno}    '
        #       f'Class: {".".join([x.node.name for x in self.class_def])}, Function: {self.function_def.name}')
        for stmt in self.call_statements[i: j]:
            print(f'{self.function_def.lineno + stmt.ast_node.lineno - 1}    '
                  f'{astunparse.unparse(stmt.ast_node).strip()}')
                  # f'{".".join(stmt.refactoredFunctionId)+"@" if stmt.refactoredFunctionId is not None else ""}'
                  # f'{".".join(stmt.functionId)}({[parse_attribute(x) if x is not None else "" for x in stmt.arguments] if stmt.arguments is not None else ""}, '
                  # f'{[x for x in stmt.keywords] if stmt.keywords is not None else ""})'
                  # f'{(" -> " + parse_attribute(stmt.output)) if stmt.output is not None else ""}')


class classInfo(object):
    def __init__(self, node, func_dict, class_dict):
        self.node = node
        self.functions = func_dict
        self.subclasses = class_dict
        self.attributes = {}
        self.variable_def = {}

    def set_attribute(self, attr, node):
        self.attributes[attr] = node

    def set_variable(self, func_name, var, node):
        if func_name not in self.variable_def:
            self.variable_def[func_name] = {}
        self.variable_def[func_name][var] = node

    def look_up_attribute(self, attr):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return None

    def look_up_variable(self, func_name, var):
        if func_name not in self.variable_def:
            return None
        elif var in self.variable_def[func_name]:
            return self.variable_def[func_name][var]
        else:
            return None


class moduleVisitor(ast.NodeVisitor):
    def __init__(self, module):
        self.module_name = module
        self.module_path = module.replace('.', '/') + '.py'
        self.packageMngr = packageMngr()
        self.global_var = {}
        self.global_func_var_def = {}
        tc = importlib.import_module(module)
        self.dict = tc.__dict__
        self.call_sequences = []
        self.call_sequences_in_class = []
        self.class_def_stack = []
        self.current_func = None
        self.temp_var_index = 0
        self.tmp_call_sequence = None

        with open(self.module_path) as f:
            self.ast_node = ast.parse(f.read())

    def get_paths(self, graph):
        init_node = None
        for block in graph.blocks:
            if block.label is not None and block.label.startswith('<'):
                continue
            if not block.control_flow_nodes or len(block.control_flow_nodes) == 0: continue
            init_node = block.control_flow_nodes[0]
            break
        if not init_node:
            return None
        return self._get_paths(init_node)

    def _get_paths(self, node):
        if len(node.next) == 0:
            return [[node]]
        paths = []
        for next_node in node.next:
            for path in self._get_paths(next_node):
                paths.append([node] + path)
        return paths

    def get_blocks(self, graph):
        blocks = []
        for block in graph.blocks:
            if block.label is not None and block.label.startswith('<'):
                continue
            if not block.control_flow_nodes or len(block.control_flow_nodes) == 0:
                continue
            blocks.append(block.control_flow_nodes)
        return blocks

    def get_temp_variable(self):
        var_name = f'temp_{self.temp_var_index}'
        self.temp_var_index += 1
        return var_name

    def begin_visit(self):
        self.visit(self.ast_node)

    def visit_Import(self, node):
        self.packageMngr.register(node)

    def visit_ImportFrom(self, node):
        self.packageMngr.register(node)

    def visit_Assign(self, node):
        if len(self.class_def_stack) != 0:
            class_info = self.class_def_stack[-1]
            class_info.set_variable('__self__', unparse(node.targets[0]), node.value)
            return
        self.global_var[unparse(node.targets[0])] = node.value

    def visit_ClassDef(self, node):
        class_name = node.name
        if len(self.class_def_stack) == 0:
            class_inspectee = self.dict[class_name]
        else:
            class_inspectee = self.class_def_stack[-1].subclasses[class_name]
        func_dict = {}
        class_dict = {}
        for fname, fn in inspect.getmembers(class_inspectee, predicate=inspect.isfunction):
            func_dict[fname] = fn
        for cname, cls in inspect.getmembers(class_inspectee, predicate=inspect.isclass):
            class_dict[cname] = cls
        self.class_def_stack.append(classInfo(node, func_dict, class_dict))
        self.call_sequences_in_class = []
        self.generic_visit(node)
        self.call_sequences.append(self.call_sequences_in_class)
        self.class_def_stack.pop()

    def visit_FunctionDef(self, node):
        function_name = node.name
        if len(self.class_def_stack) != 0 and function_name in self.class_def_stack[-1].functions:
            function_inspector = self.class_def_stack[-1].functions[function_name]
        else:
            return
            function_inspector = self.dict[function_name]
        graph = control_flow.get_control_flow_graph(function_inspector)
        self.current_func = node
        paths = self.get_paths(graph)
        blocks = self.get_blocks(graph)
        if blocks is None:
            self.current_func = None
            return

        for path in paths:
            self.tmp_call_sequence = callSequence(self)
            for control_flow_node in path:
                instruction_node = control_flow_node.instruction.node
                ast_node = gast.gast_to_ast(instruction_node)
                call_visitor = callVisitor(ast_node, self)
                call_visitor.visit(ast_node)
            if len(self.tmp_call_sequence.call_statements) > 0:
                self.call_sequences_in_class.append(self.tmp_call_sequence)
                # self.tmp_call_sequence.print()

            pass

        self.packageMngr.clear_tmp()
        self.current_func = None
        return


if __name__ == '__main__':
    test = moduleVisitor('higginface_transformers.src.transformers.models.bert.modeling_bert')
    test.begin_visit()
