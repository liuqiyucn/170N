import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import itertools

# this is the function definition cell
def volume_of_cylinder(h,r):
    S=3.14159*r**2
    V=S*h
    return V

def recursive_factorial(n : int) -> int:
    if n < 0:
        raise Exception('n cannot be negative')
    if n == 1:
        return 1
    if n == 0 :
        return 1
    else:
        return n*recursive_factorial(n-1)
    
fib_memo = {}

def F_memo(n):
    if n == 1:
        return 0
    elif n == 2:
        return 1
    elif n not in fib_memo :
        fib_memo[n] = F_memo(n - 1) + F_memo(n - 2)
    return fib_memo[n]

def F(n):
    if n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        return F(n - 1) + F(n - 2)
    
def recursive_factorial_memo(n):
    if n < 0:
        raise Exception('n cannot be negative')
    if n == 0:
        return 1
    elif n == 1:
        return 1
    elif n not in fac_memo :
        fac_memo[n] = n * recursive_factorial_memo(n - 1)
    return fac_memo[n]

fac_memo = {}

def binomial_factorial(n: int, k: int) -> int:
    if n < 0 or k < 0:
        raise Exception('n or k cannot be negative')
    if k > n:
        return 0
        # raise exception here
    return recursive_factorial(n)/(recursive_factorial(k)*recursive_factorial(n-k))

def binomial_recursive(n : int, k : int) -> int:
    if n < 0 or k < 0:
        raise Exception('n or k cannot be negative')
    if k > n:
        raise Exception('k cannot be greater than n')
    if k == 0:
        return 1
    if n == k:
        return 1
    else:
        return binomial_recursive(n-1, k-1) + binomial_recursive(n-1, k)
    
def logistic(n : int, r : float, x0 : float):
    if n < 0:
        raise Exception('n cannot be negative')
    if n == 0:
        return r*x0*(1-x0)
    else:
        term = logistic(n-1, r, x0)
        return r*term*(1-term)
    
def linear_search(L : list[int], n : int):
    for i in range(len(L)):
        if n == L[i]:
            return i
    print('The specific number is not in the list.')
    return 

def bisection_search(L : list[int], n : int):
    start = 0
    end = len(L) -1
    middle = 0
    while (end-start+1) > 2:
        middle = int((start + end)/2)
        if L[middle] < n:
            start = middle
        else:
            end = middle
    if L[start] == n or L[end] == n:
        return True
    else:
        return False
    

def bisection_root(f, x_left : float, x_right : float, epsilon : float):
    while (x_right - x_left) / 2 > epsilon:
        mid = (x_right + x_left)/2
        
        if f(mid) == 0:
            return mid
        
        if f(x_left) * f(mid) < 0:
            x_right = mid
        else:
            x_left = mid
    
    return (x_left + x_right)/2

@njit
def sieve_Eratosthenes(n : int) -> list[int]:
    if n < 2:
        raise Exception('n must be greater or equal to 2')
    if n == 2:
        return [2]
    numbers = [i for i in range(n+1)]
    isPrime = [True for i in range(n+1)]
    prime = 2
    while (prime**2 <= n):
        if (isPrime[prime] == True):
            for i in range(prime**2, n+1):
                if i % prime == 0:
                    isPrime[i] = False
        prime = prime + 1
    
    delete = [0, 1]
    numbers = np.array(numbers)
    isPrime = np.array(isPrime)
    return list(np.delete(numbers[isPrime], delete))

def sieve_Eratosthenes_nonoptimized(n : int) -> list[int]:
    if n < 2:
        raise Exception('n must be greater or equal to 2')
    if n == 2:
        return [2]
    numbers = [i for i in range(n+1)]
    isPrime = [True for i in range(n+1)]
    prime = 2
    while (prime**2 <= n):
        if (isPrime[prime] == True):
            for i in range(prime**2, n+1):
                if i % prime == 0:
                    isPrime[i] = False
        prime = prime + 1
    
    delete = [0, 1]
    numbers = np.array(numbers)
    isPrime = np.array(isPrime)
    return list(np.delete(numbers[isPrime], delete))

def prime_factors(n : int) -> list[int]:
    if n < 2:
        raise Exception('n should be greater or equal than 2')
    prime_list = sieve_Eratosthenes(n)
    out = []
    prime_index = 0
    while n >= 2:
        if n % prime_list[prime_index] == 0:
            out.append(prime_list[prime_index])
            n = n / prime_list[prime_index]
        else:
            prime_index = prime_index + 1
    return out

@njit
def nth_prime(n : int) -> int:
    if n < 1:
        raise Exception('input cannot be less than 1')
    elif n == 1:
        return 2
    elif n == 2:
        return 3
    
    guess_N = int(n * (np.log(n) + np.log(np.log(n))))
    prime_list = sieve_Eratosthenes(guess_N)
    while len(prime_list) < n:
        guess_N = guess_N * 2
        prime_list = sieve_Eratosthenes(guess_N)
    return prime_list[n-1]

def nth_prime_non_optimized(n : int) -> int:
    if n < 1:
        raise Exception('input cannot be less than 1')
    elif n == 1:
        return 2
    elif n == 2:
        return 3
    
    guess_N = int(n * (np.log(n) + np.log(np.log(n))))
    prime_list = sieve_Eratosthenes_nonoptimized(guess_N)
    while len(prime_list) < n:
        guess_N = guess_N * 2
        prime_list = sieve_Eratosthenes_nonoptimized(guess_N)
    return prime_list[n-1]

def top_sum(side: int, top_num : int, total_dice_num : int, target : int) -> int:
    # how many ways can 10 dice sum to n
    
    return 0