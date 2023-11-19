import sys
import numpy as np
import scipy.optimize as opt

def main():
    """
    Для чтения входных данных необходимо получить их
    из стандартного потока ввода (sys.stdin).
    Данные во входном потоке соответствуют описанному
    в условии формату. Обычно входные данные состоят
    из нескольких строк.
    Можно использовать несколько методов:
    * input() -- читает одну строку из потока без символа
    перевода строки;
    * sys.stdin.readline() -- читает одну строку из потока,
    сохраняя символ перевода строки в конце;
    * sys.stdin.readlines() -- вернет список (list) строк,
    сохраняя символ перевода строки в конце каждой из них.
    Чтобы прочитать из строки стандартного потока:
    * число -- int(input()) # в строке должно быть одно число
    * строку -- input()
    * массив чисел -- map(int, input().split())
    * последовательность слов -- input().split()
    Чтобы вывести результат в стандартный поток вывода (sys.stdout),
    можно использовать функцию print() или sys.stdout.write().
    Возможное решение задачи "Вычислите сумму чисел в строке":
    print(sum(map(int, input().split())))
    """
    l = []
    n = int(input())
    for line in sys.stdin.readlines():
        l.append(int(line))
    
    l = np.array(l)
    
    m = np.mean(l)
    print(m)
    print(np.median(l))
    
    def func(x):
        x = x[0]
        return np.abs((l - x) / l).sum() / n
    
    a = opt.minimize(func, [m], tol=1e-12, method='Nelder-Mead')
    
    print(a.x[0])


if __name__ == '__main__':
    main()