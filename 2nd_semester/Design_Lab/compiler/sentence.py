import ply.lex as lex
import ply.yacc as yacc

tokens = ['WORDS','VERBS']


def t_VERBS(t):
    r'(is|are|am)'
    return t  

def t_WORDS(t):
    r'[A-Za-z]+'
    return t
                                                                                                                                                            

t_ignore = ' \t\n'

def t_error(t):
    print('Lexical error')
    t.lexer.skip(1)

def p_sentence(p):
    '''sentence : WORDS VERBS sentence
                | sen1'''
    if len(p)==4:
        p[0] =" " + p[1] + " " + p[2] + " " + p[3]
    else:
        p[0] = " " + p[1]

def p_sen1(p):
    '''sen1 : WORDS sen1
            | WORDS'''
    if len(p)==3 :
        p[0] = p[1] + " " + p[2]
    else:
        p[0] = " " + p[1]
def p_error(p):
    print("Error")
    #print(p.value)
    raise SyntaxError
lexer = lex.lex()
parser = yacc.yacc()

while True:
    sentence = input("Sentence : ")
    try:
        result = parser.parse(sentence)
    except SyntaxError:
        result= None
    if result is not None:
        print(result.lstrip())