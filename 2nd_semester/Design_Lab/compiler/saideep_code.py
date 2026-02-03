import ply.lex as lex
import ply.yacc as yacc

# --- Tokens ---
tokens = (
    "LABEL",  # e.g., L0, L1
    "DOLLAR",  # $$$
    "ID",  # Identifiers (e.g., registers A, B, etc.)
    "MEMREF",
    "NUMBER",
    "STRING",
    "COMMA",
    "EQ", "NE", "GT", "LT",  # Relational operators
    "STOR", "SUM", "SUB", "MUL", "DIV", "MOD",  # Arithmetic keywords
    "AND", "OR", "XOR", "NOT", "SHL", "SHR",  # Logical keywords
    "IF", "GOTO", "HLT", "PRINT",
    "CONCAT", "LENGTH", "SUBSTR",  # String operations
)

# --- Regular Expressions ---
t_DOLLAR = r'\$\$\$'
t_COMMA = r','
t_EQ = r'=='
t_NE = r'!='
t_GT = r'>'
t_LT = r'<'

def t_LABEL(t):
    r'[Ll]\d+'
    return t

def t_MEMREF(t):
    r'@[a-zA-Z_][a-zA-Z0-9_]*'
    return t

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value.upper() in keyword_map:
        t.type = t.value.upper()
    return t

def t_NUMBER(t):
    r'-?\d+(\.\d+)?'
    t.value = float(t.value) if '.' in t.value else int(t.value)
    return t

def t_STRING(t):
    r'"([^\\"]|(\\"))*"'
    t.value = t.value[1:-1]
    return t

t_ignore = ' \t'

# Map keywords to token types
keyword_map = {
    'STOR': 'STOR', 'SUM': 'SUM', 'SUB': 'SUB', 'MUL': 'MUL', 'DIV': 'DIV', 'MOD': 'MOD',
    'AND': 'AND', 'OR': 'OR', 'XOR': 'XOR', 'NOT': 'NOT', 'SHL': 'SHL', 'SHR': 'SHR',
    'IF': 'IF', 'GOTO': 'GOTO', 'HLT': 'HLT', 'PRINT': 'PRINT',
    'CONCAT': 'CONCAT', 'LENGTH': 'LENGTH', 'SUBSTR': 'SUBSTR'
}

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_error(t):
    print(f"LEX ERROR: Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# --- Parser ---
def p_program(p):
    'program : instruction_list'
    p[0] = p[1]

def p_instruction_list(p):
    '''instruction_list : instruction instruction_list
                        | instruction'''
    if len(p) == 3:
        p[0] = [p[1]] + p[2]
    else:
        p[0] = [p[1]]

def p_instruction(p):
    'instruction : opt_label command'
    if p[1]:  # If there is a label
        p[2]['label'] = p[1]
    p[0] = p[2]

def p_opt_label(p):
    '''opt_label : LABEL DOLLAR
                 | empty'''
    if len(p) == 3:
        p[0] = p[1]
    else:
        p[0] = None

def p_command(p):
    '''command : memory_command
               | arithmetic_command
               | logical_command
               | control_flow_command
               | print_command
               | string_command'''
    p[0] = p[1]

def p_memory_command(p):
    'memory_command : STOR operand COMMA operand'
    p[0] = {'opcode': 'STOR', 'operands': [p[2], p[4]]}

def p_arithmetic_command(p):
    '''arithmetic_command : SUM operand COMMA operand
                          | SUB operand COMMA operand
                          | MUL operand COMMA operand
                          | DIV operand COMMA operand
                          | MOD operand COMMA operand'''
    p[0] = {'opcode': p[1].upper(), 'operands': [p[2], p[4]]}

def p_logical_command(p):
    '''logical_command : AND operand COMMA operand
                       | OR operand COMMA operand
                       | XOR operand COMMA operand
                       | NOT operand
                       | SHL operand COMMA operand
                       | SHR operand COMMA operand'''
    if len(p) == 3:  # Unary operation (e.g., NOT)
        p[0] = {'opcode': p[1].upper(), 'operands': [p[2]]}
    else:  # Binary operation
        p[0] = {'opcode': p[1].upper(), 'operands': [p[2], p[4]]}

def p_control_flow_command(p):
    '''control_flow_command : IF condition command
                            | GOTO LABEL
                            | HLT'''
    if p[1].upper() == 'IF':
        p[0] = {'opcode': 'IF', 'condition': p[2], 'then': p[3]}
    elif p[1].upper() == 'GOTO':
        p[0] = {'opcode': 'GOTO', 'label_target': p[2]}
    elif p[1].upper() == 'HLT':
        p[0] = {'opcode': 'HLT'}

def p_print_command(p):
    'print_command : PRINT operand'
    p[0] = {'opcode': 'PRINT', 'operands': [p[2]]}

def p_string_command(p):
    '''string_command : CONCAT operand COMMA operand
                      | LENGTH operand COMMA operand
                      | SUBSTR operand COMMA operand COMMA operand'''
    if p[1].upper() == 'CONCAT':
        p[0] = {'opcode': 'CONCAT', 'operands': [p[2], p[4]]}
    elif p[1].upper() == 'LENGTH':
        p[0] = {'opcode': 'LENGTH', 'operands': [p[2], p[4]]}
    elif p[1].upper() == 'SUBSTR':
        p[0] = {'opcode': 'SUBSTR', 'operands': [p[2], p[4], p[6]]}

def p_condition(p):
    '''condition : operand EQ operand
                 | operand NE operand
                 | operand GT operand
                 | operand LT operand'''
    p[0] = (p[1], p[2], p[3])

def p_operand(p):
    '''operand : NUMBER
               | STRING
               | MEMREF
               | ID'''
    p[0] = p[1]

def p_empty(p):
    'empty :'
    pass

def p_error(p):
    if p:
        print(f"PARSE ERROR: Unexpected token '{p.value}' at line {p.lineno}")
    else:
        print("PARSE ERROR: End of input")

# Build the parser
parser = yacc.yacc()

# --- Interpreter ---
registers = {}
memory = {}

def get_value(operand):
    if isinstance(operand, (int, float)):
        return operand  # Numeric literal
    if isinstance(operand, str):
        if operand.startswith('@'):
            return memory.get(operand[1:], None)  # Memory reference
        elif operand in registers:
            return registers.get(operand, None)  # Register lookup
    return operand  # Literal strings or unhandled types

def set_value(operand, value):
    if operand.startswith('@'):
        memory[operand[1:]] = value
    else:
        registers[operand] = value

def evaluate_condition(cond):
    left, op, right = cond
    left_val = get_value(left)
    right_val = get_value(right)
    return {
        '==': left_val == right_val,
        '!=': left_val != right_val,
        '>': left_val > right_val,
        '<': left_val < right_val
    }.get(op, False)

def run_program(instructions):
    labels = {instr['label']: i for i, instr in enumerate(instructions) if instr.get('label')}
    pc = 0
    while pc < len(instructions):
        instr = instructions[pc]
        op = instr['opcode']

        if op == 'HLT':
            print("\nFinal Registers:", registers)
            print("Final Memory:", memory)
            break
        elif op == 'STOR':
            dest, src = instr['operands']
            resolved_value = get_value(src)
            set_value(dest, resolved_value)
        elif op in ('SUM', 'SUB', 'MUL', 'DIV', 'MOD'):
            left, right = instr['operands']
            lval, rval = get_value(left), get_value(right)
            if op in ('DIV', 'MOD') and rval == 0:
                print(f"RUNTIME ERROR: Division by zero in operation '{op} {left}, {right}'")
                registers[left] = None
                pc += 1
                continue
            result = {
                'SUM': lval + rval,
                'SUB': lval - rval,
                'MUL': lval * rval,
                'DIV': lval / rval,
                'MOD': lval % rval
            }[op]
            set_value(left, result)
        elif op == 'PRINT':
            value = instr['operands'][0]
            print(get_value(value))
        elif op == 'GOTO':
            if instr['label_target'] in labels:
                pc = labels[instr['label_target']]
                continue
        elif op == 'IF':
            if evaluate_condition(instr['condition']):
                run_program([instr['then']])
        elif op == 'CONCAT':
            left, right = instr['operands']
            result = str(get_value(left)) + str(get_value(right))
            set_value(left, result)
        elif op == 'LENGTH':
            dest, src = instr['operands']
            result = len(str(get_value(src)))
            set_value(dest, result)
        elif op == 'SUBSTR':
            dest, start, end = instr['operands']
            result = str(get_value(dest))[int(get_value(start)):int(get_value(end))]
            set_value(dest, result)
        elif op in ('AND', 'OR', 'XOR', 'NOT'):
            if op == 'NOT':
                operand = instr['operands'][0]
                result = ~int(get_value(operand))
                set_value(operand, result)
            else:
                left, right = instr['operands']
                lval = int(get_value(left))
                rval = int(get_value(right))
                result = {
                    'AND': lval & rval,
                    'OR': lval | rval,
                    'XOR': lval ^ rval
                }[op]
                set_value(left, result)
        pc += 1

# --- Main Program ---
def main():
    print("Enter your program (end with an empty line):")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line.strip())
    source = "\n".join(lines)
    instructions = parser.parse(source, lexer=lexer)
    if instructions:
        run_program(instructions)
    else:
        print("Parsing failed.")

if __name__ == "__main__":
    main()
