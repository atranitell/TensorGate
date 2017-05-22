
class A():

    def __init__(self):
        print('a')

    def a(self):
        print('a.a')

class B(A):

    def __init__(self):
        print('b')

    def a(self):
        print('b.a')

b = B()
b.a()