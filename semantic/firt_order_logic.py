# -*- coding: utf-8 -*-
# Author Zhang Chen
# Email ZhangChen.Shaanxi@gmail.com
# Date 2018/11/22

import nltk

read_expr = nltk.sem.Expression.fromstring
expr = read_expr('exists x.(dog(x) & disappear(x))')
print(expr)

dom = {'b', 'o', 'c'}
v = """
bertie => b
olive => o
cyril => c
boy => {b}
girl => {o}
dog => {c}
walk => {o, c}
see => {(b, o), (c, b), (o, c)}
 """

val = nltk.Valuation.fromstring(v)

print(val)

print(('o', 'c') in val['see'])

g = nltk.Assignment(dom, [('x', 'o'), ('y', 'c')])
print(g['y'])

m = nltk.Model(dom, val)
print(m.evaluate('see(olive, y)', g))

print(g)
g.purge()
print(g)

a4 = read_expr('exists y. (woman(y) & all x. (man(x) -> love(x,y)))')
a5 = read_expr('man(adam)')
a6 = read_expr('woman(eve)')
g = read_expr('love(adam,eve)')
mc = nltk.MaceCommand(g, assumptions=[a4, a5, a6])
print(mc.build_model())
print(mc.valuation)

read_dexpr = nltk.sem.DrtExpression.fromstring
drs1 = read_dexpr('([x, y], [angus(x), dog(y), own(x, y)])')
print(drs1)
#drs1.draw()

dt = nltk.DiscourseTester(['A student dances', 'Every student is a person'])
dt.readings()
dt.add_sentence('No person dances', consistchk=True)
dt.retract_sentence('No person dances', verbose=True)
dt.add_sentence('A person dances', informchk=True)