'''#define function
def getLarg(n1,n2):
    Compare two Number
    if(n1>n2):
        print("{} is greater".format(n1))
        return n1
    else:
        print("{} is greater".format(n2))
        return n2
#call function
getLarg(12,34)'''
'''
def sumtwo():
    a=int(input("Num1 :"))
    b=int(input("Num2 :"))
    c=a+b
    return c
x=sumtwo()

print("Sum is :",x)'''

'''def isPrime(n):
    for x in range(2,n):
        if(n%x==0):
            return False
    else:
        return True



def primeSerise():
    a=int(input("Min :"))
    b=int(input("Max :"))
    print ("Prime numbers between{} and {} are :".format(a,b))
    for x in range(a,b+1):
        if(isPrime(x)):
            print(x,end=', ')
primeSerise()'''

#Example of default argument
'''
import math as m
def calArea(n1,n2=1):
    str =""
    if(n2==1):
        area = m.pi*n1*n1
        str = "area of Circle is {}".format(area)
        return str
    else:
        area = n1*n2
        str = "area of Rectangle is {}".format(area)
        return str

print(calArea(12))

print(calArea(12,15))'''


#Example of Arbitrary Argument
def calAvg(*n):
    s=0
    c=0
    for x in n:
        s=s+x
        c +=1
    avg = s/c
    print("avg :",avg)
    
calAvg(2,3,4)
calAvg(4,5,6,7,8,9)
        


        
        

