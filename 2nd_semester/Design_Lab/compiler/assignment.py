import ply.lex as lex
import ply.yacc as yacc

# Lexer
tokens = [
    'LABEL', 'DOLLAR', 'OPCODE', 'REG', 'NUMBER', 'STRING', 'COMMA', 'COMPARISON', 'IF', 'GOTO',
]

t_DOLLAR = r'\$\$\$'
t_COMPARISON = r'==|!=|<=|>=|<|>'
t_COMMA = r','
t_ignore = ' \t'

def t_OPCODE(t):
    r'STOR|PRINT|HLT|SUM|MUL|DIV|MOD|AND|OR|XOR|NOT|SHL|SHR|CONCAT|LENGTH|SUBSTR'
    return t

def t_GOTO(t):
    r'GOTO' 
    return t

def t_IF(t):
    r'IF'
    return t

def t_LABEL(t):
    r'L\d+'
    return t

def t_REG(t):
    r'@?[A-Za-z]'
    return t

def t_STRING(t):
    r'"[^"]*"'
    t.value = t.value.strip('"')
    return t

def t_NUMBER(t):
    r'-?\d+(\.\d+)?'
    t.value = float(t.value) if '.' in t.value else int(t.value)
    return t

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"Illegal character {t.value[0]} at line {t.lexer.lineno}")
    t.lexer.skip(1)

# Parser
instruction = []

def p_instructions(p):
    '''instructions : instructions instruction
                    | instruction'''
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
        instruction.append(p[2])
    else:
        p[0] = [p[1]]
        instruction.append(p[1])

def p_instruction(p):
    '''instruction : LABEL DOLLAR OPCODE operands
                   | LABEL DOLLAR GOTO LABEL
                   | condition'''
    if len(p) == 5:
        p[0] = {'label': p[1], 'opcode': p[3], 'operands': p[4]}
    elif len(p) == 4:
        p[0] = {'label': p[1], 'opcode': p[3], 'operands': p[4]}
    else:
        p[0] = p[1]

def p_condition(p):
    '''condition : LABEL DOLLAR IF comparison GOTO LABEL
                 | LABEL DOLLAR IF comparison OPCODE operands'''
    p[0] = {
        'label': p[1],
        'opcode': p[3],
        'comparison': p[4],
        'operation': {
            'opcode': p[5], 'operands': p[6]
        }
    }

def p_comparison(p):
    '''comparison : operand COMPARISON operand'''
    p[0] = (p[1], p[2], p[3])

def p_operands(p):
    '''operands : operand COMMA operands
                | operand'''
    p[0] = [p[1]] + p[3] if len(p) == 4 else [p[1]]

def p_operand(p):
    '''operand : REG
               | NUMBER
               | STRING'''
    p[0] = p[1]

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' on line {p.lineno}")
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()

# Virtual Machine
class VirtualMachine:
    def __init__(self):
        self.registers = {}
        self.print_statements = []

    def resolve_value(self, operand):
        if isinstance(operand, str) and operand in self.registers:
            return self.registers[operand]
        return operand

    def run_instruction(self, inst, instructions):
        opcode = inst['opcode']
        operands = inst.get('operands', [])
        if opcode == 'STOR':
            self.stor(operands)
        elif opcode == 'SUM':
            self.arithmetic_command(operands, lambda a, b: a + b)
        elif opcode == 'MUL':
            self.arithmetic_command(operands, lambda a, b: a * b)
        elif opcode == 'DIV':
            self.arithmetic_command(operands, lambda a, b: a / b if b != 0 else print("Error: Division by zero"))
        elif opcode == 'MOD':
            self.arithmetic_command(operands, lambda a, b: a % b if b != 0 else print("Error: Division by zero"))
        elif opcode == 'AND':
            self.bitwise_command(operands, lambda a, b: a & b)
        elif opcode == 'OR':
            self.bitwise_command(operands, lambda a, b: a | b)
        elif opcode == 'XOR':
            self.bitwise_command(operands, lambda a, b: a ^ b)
        elif opcode == 'NOT':
            self.not_command(operands)
        elif opcode == 'SHL':
            self.shift_command(operands, lambda a, b: a << b)
        elif opcode == 'SHR':
            self.shift_command(operands, lambda a, b: a >> b)
        elif opcode == 'PRINT':
            self.print_register(operands)
        elif opcode == 'CONCAT':
            self.concat_command(operands)
        elif opcode == 'LENGTH':
            self.length_command(operands)
        elif opcode == 'SUBSTR':
            self.substr_command(operands)
        elif opcode == 'IF':
            a, op, b = inst['comparison']
            a = self.resolve_value(a)
            b = self.resolve_value(b)
            if self.compare(a, op, b):
                return self.run_instruction(inst['operation'], instructions)
        elif opcode == 'GOTO':
            return self.goto(operands, instructions)
        elif opcode == 'HLT':
            print("Halting execution.")
            return -1
        else:
            print(f"Unknown opcode: {opcode}")
        return instructions.index(inst)

    def execute(self, instructions):
        size = len(instructions)
        i = 0
        while i < size:
            i = self.run_instruction(instructions[i], instructions)
            if i == -1:
                break
            i += 1

    def compare(self, a, op, b):
        a = self.registers[a] if a in self.registers else a
        b = self.registers[b] if b in self.registers else b
        return {
            '==': a == b,
            '!=': a != b,
            '<': a < b,
            '>': a > b,
            '<=': a <= b,
            '>=': a >= b,
        }.get(op, False)

    def goto(self, label, instructions):
        for idx, inst in enumerate(instructions):
            if inst['label'] == label:
                return idx - 1
        print(f"Error: Label {label} not found.")
        return len(instructions)

    def stor(self, operands):
        reg = operands[0]
        value = self.resolve_value(operands[1])
        self.registers[reg] = value

    def arithmetic_command(self, operands, operation):
        reg = operands[0]
        if reg not in self.registers:
            print(f"Error: Register {reg} not initialized.")
            return
        value = self.resolve_value(operands[1])
        self.registers[reg] = operation(self.registers[reg], value)

    def bitwise_command(self, operands, operation):
        reg = operands[0]
        if reg not in self.registers:
            print(f"Error: Register {reg} not initialized.")
            return
        value = self.resolve_value(operands[1])
        
        # Ensure operands are integers before performing bitwise operation
        self.registers[reg] = operation(int(self.registers[reg]), int(value))


    def not_command(self, operands):
        reg = operands[0]
        if reg not in self.registers:
            print(f"Error: Register {reg} not initialized.")
            return
        self.registers[reg] = ~self.registers[reg]

    def shift_command(self, operands, operation):
        reg = operands[0]
        if reg not in self.registers:
            print(f"Error: Register {reg} not initialized.")
            return
        value = self.resolve_value(operands[1])
        self.registers[reg] = operation(self.registers[reg], value)

    def concat_command(self, operands):
        reg_a, reg_b = operands
        if reg_a in self.registers and reg_b in self.registers:
            self.registers[reg_a] += self.registers[reg_b]
        else:
            print(f"Error: Registers {reg_a} or {reg_b} not initialized.")

    def length_command(self, operands):
        reg_a, reg_b = operands
        if reg_b in self.registers and isinstance(self.registers[reg_b], str):
            self.registers[reg_a] = len(self.registers[reg_b])
        else:
            print(f"Error: Register {reg_b} not initialized or does not contain a string.")

    def substr_command(self, operands):
        reg_a, pos1, pos2 = operands
        pos1 = self.resolve_value(pos1)
        pos2 = self.resolve_value(pos2)
        if reg_a in self.registers and isinstance(self.registers[reg_a], str):
            self.registers[reg_a] = self.registers[reg_a][pos1:pos2]
        else:
            print(f"Error: Register {reg_a} not initialized or does not contain a string.")

    def print_register(self, operands):
        operand = operands[0]
        if isinstance(operand, str) and operand in self.registers:
            self.print_statements.append(f"Value in register {operand}: {self.registers[operand]}")
        elif isinstance(operand, str):
            self.print_statements.append(operand)

    def print_final_registers(self):
        print("\nFinal Register Values:")
        for reg, value in self.registers.items():
            print(f"{reg}: {value}")
        if self.print_statements:
            print("\nPrint Statements:")
            for statement in self.print_statements:
                print(statement)


# Example program with string operations
program = """
L0 $$$ STOR A, "Study the examples"
L1 $$$ STOR a, 10
L2 $$$ STOR b, 2
L3 $$$ STOR @b, 35
L4 $$$ SUM a, 5
L5 $$$ PRINT a
L6 $$$ IF a <= 100 GOTO L4
L7 $$$ PRINT "Done"
L8 $$$ GOTO L9
L9 $$$ PRINT "Unreachable"
L10 $$$ STOR C, "End"
L11 $$$ PRINT C
L12 $$$ HLT
"""
lexer = lex.lex()
# lexer.input(program)
# for tok in lexer:
#     print(tok)

parser.parse(program)
# print(instruction)

if instruction:
    print("\nExecuting Program:")
    vm = VirtualMachine()
    vm.execute(instruction)
    vm.print_final_registers()
else:
    print("No instructions to execute.")
