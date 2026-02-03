import ply.lex as lex
import ply.yacc as yacc
import sys

tokens = ['PARLEFT', 'PARRIGHT']

t_PARLEFT = r'\('  # Token for left paranthesis
t_PARRIGHT = r'\)' # Token for Right paranthesis

t_ignore = r' '  # Ignore Spaces

def t_error(t):   # Skip Illegal Characters
    print("InValid Character found: {}".format(t.value[0]))
    t.lexer.skip(1)

lexer = lex.lex()

def p_expS(p):
   '''
   expS : PARLEFT PARRIGHT
        | PARLEFT expA
        | expS expS
   '''
   p[0] = (p[1] , p[2])

def p_expA(p):
    '''
    expA : expS PARRIGHT
    '''
    p[0] = (p[0] , p[1] , p[2])

def p_error(p):
    if not p:
        print("Syntax Error Found @ EOF")
    else:
        print("Syntax Error Found @ {}".format(p.lexpos))

parser = yacc.yacc()

def display_lexer_tokens(str_exp):
    while True:
      LToken = lexer.token()
      if not LToken:
         break
      print(LToken)

def check_balanced_paranthesis(str_exp):
    checkFlag = parser.parse(str_exp)
    print()
    if checkFlag is None:
        print("Paranthesis are Unbalanced...")
    else:
        print("Paranthesis are Balanced...")

input_str = input("Enter String: ")
lexer.input(input_str)
print()
display_lexer_tokens(input_str)
print()
check_balanced_paranthesis(input_str)