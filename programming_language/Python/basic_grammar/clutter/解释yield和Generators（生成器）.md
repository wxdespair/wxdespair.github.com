转自：http://www.oschina.net/translate/improve-your-python-yield-and-generators-explained

原文：http://www.jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/
  		

在开始课程之前，我要求学生们填写一份调查表，这个调查表反映了它们对 Python 中一些概念的理解情况。一些话题（“if/else 控制流”或者“定义和使用函数”）对于大多数学生是没有问题的。但是有一些话题，大多数学生只有很少，或者完全没有任何接触，尤其是“生成器和 yield 关键字”。我猜这对大多数新手 Python 程序员也是如此。

有事实表明，在我花了大功夫后，有些人仍然不能理解生成器和 yield 关键字。我想让这个问题有所改善。在这篇文章中，我将解释 yield 关键字到底是什么，为什么它是有用的，以及如何来使用它。

*注意：最近几年，生成器的功能变得越来越强大，它已经被加入到了 PEP。在我的下一篇文章中，我会通过协程（coroutine），协同式多任务处理（cooperative multitasking），以及异步 IO（asynchronous I/O）（尤其是 GvR 正在研究的“tulip”原型的实现)来介绍 yield 的真正威力。但是在此之前，我们要对生成器和 yield 有一个扎实的理解。*

## **协程（协同程序）与子例程**

我们调用一个普通的 Python 函数时，一般是从函数的第一行代码开始执行，结束于 return 语句、异常或者函数结束（可以看作隐式的返回 None）。一旦函数将控制权交还给调用者，就意味着全部结束。函数中做的所有工作以及保存在局部变量中的数据都将丢失。再次调用这个函数时，一切都将从头创建。 

对于在计算机编程中所讨论的函数，这是很标准的流程。这样的函数只能返回一个值，不过，有时可以创建能产生一个序列的函数还是有帮助的。要做到这一点，这种函数需要能够“保存自己的工作”。 

我说过，能够“产生一个序列”是因为我们的函数并没有像通常意义那样返回。return 隐含的意思是函数正将执行代码的控制权返回给函数被调用的地方。而 yield 的隐含意思是控制权的转移是临时和自愿的，我们的函数将来还会收回控制权。

在 Python 中，拥有这种能力的“函数”被称为生成器，它非常的有用。生成器（以及 yield 语句）最初的引入是为了让程序员可以更简单的编写用来产生值的序列的代码。 以前，要实现类似随机数生成器的东西，需要实现一个类或者一个模块，在生成数据的同时保持对每次调用之间状态的跟踪。引入生成器之后，这变得非常简单。

为了更好的理解生成器所解决的问题，让我们来看一个例子。在了解这个例子的过程中，请始终记住我们需要解决的问题：**生成值的序列**。

*注意：在 Python 之外，最简单的生成器应该是被称为协程（coroutines）的东西。在本文中，我将使用这个术语。请记住，在 Python 的概念中，这里提到的协程就是生成器。Python 正式的术语是生成器；协程只是便于讨论，在语言层面并没有正式定义。*

### **例子：有趣的素数**

假设你的老板让你写一个函数，输入参数是一个 int 的 list，返回一个可以迭代的包含素数 1 的结果。

记住，迭代器（Iterable） 只是对象每次返回特定成员的一种能力。

你肯定认为"这很简单"，然后很快写出下面的代码：  

```python
def get_primes(input_list):
    result_list = list()
    for element in input_list:
        if is_prime(element):
            result_list.append()

    return result_list

# 或者更好一些的...

def get_primes(input_list):
    return (element for element in input_list if is_prime(element))

# 下面是 is_prime 的一种实现...

def is_prime(number):
    if number > 1:
        if number == 2:
            return True
        if number % 2 == 0:
            return False
        for current in range(3, int(math.sqrt(number) + 1), 2):
            if number % current == 0:
                return False
        return True
    return False
```

上面 is_prime 的实现完全满足了需求，所以我们告诉老板已经搞定了。她反馈说我们的函数工作正常，正是她想要的。

## **处理无限序列**

噢，真是如此吗？过了几天，老板过来告诉我们她遇到了一些小问题：她打算把我们的 get_primes 函数用于一个很大的包含数字的 list。实际上，这个 list 非常大，仅仅是创建这个 list 就会用完系统的所有内存。为此，她希望能够在调用 get_primes 函数时带上一个 start 参数，返回所有大于这个参数的素数（也许她要解决 [Project Euler problem 10](https://projecteuler.net/problem=10)）。

我们来看看这个新需求，很明显只是简单的修改 get_primes 是不可能的。 自然，我们不可能返回包含从 start 到无穷的所有的素数的列表（虽然有很多有用的应用程序可以用来操作无限序列）。看上去用普通函数处理这个问题的可能性比较渺茫。

在我们放弃之前，让我们确定一下最核心的障碍，是什么阻止我们编写满足老板新需求的函数。通过思考，我们得到这样的结论：函数只有一次返回结果的机会，因而必须一次返回所有的结果。得出这样的结论似乎毫无意义；“函数不就是这样工作的么”，通常我们都这么认为的。可是，不学不成，不问不知，“如果它们并非如此呢？”

想象一下，如果 get_primes 可以只是简单返回下一个值，而不是一次返回全部的值，我们能做什么？我们就不再需要创建列表。没有列表，就没有内存的问题。由于老板告诉我们的是，她只需要遍历结果，她不会知道我们实现上的区别。

不幸的是，这样做看上去似乎不太可能。即使是我们有神奇的函数，可以让我们从 n 遍历到无限大，我们也会在返回第一个值之后卡住：  

```python
def get_primes(start):
    for element in magical_infinite_range(start):
        if is_prime(element):
            return element
```

假设这样去调用 get_primes：

```python
def solve_number_10():
    # She *is* working on Project Euler #10, I knew it!
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 2000000:
            total += next_prime
        else:
            print(total)
            return
```

  显然，在 get_primes 中，一上来就会碰到输入等于 3 的，并且在函数的第 4 行返回。与直接返回不同，我们需要的是在退出时可以为下一次请求准备一个值。

不过函数做不到这一点。当函数返回时，意味着全部完成。我们保证函数可以再次被调用，但是我们没法保证说，“呃，这次从上次退出时的第 4 行开始执行，而不是常规的从第一行开始”。函数只有一个单一的入口：函数的第 1 行代码。  

## **走进生成器**

这类问题极其常见以至于 Python 专门加入了一个结构来解决它：生成器。一个生成器会“生成”值。创建一个生成器几乎和生成器函数的原理一样简单。

一个生成器函数的定义很像一个普通的函数，除了当它要生成一个值的时候，使用 yield 关键字而不是 return。如果一个 def 的主体包含 yield，这个函数会自动变成一个生成器（即使它包含一个 return）。除了以上内容，创建一个生成器没有什么多余步骤了。

生成器函数返回生成器的迭代器。这可能是你最后一次见到“生成器的迭代器”这个术语了， 因为它们通常就被称作“生成器”。要注意的是生成器就是一类特殊的迭代器。作为一个迭代器，生成器必须要定义一些方法（method），其中一个就是 __next__()。如同迭代器一样，我们可以使用 next() 函数来获取下一个值。

为了从生成器获取下一个值，我们使用 next() 函数，就像对付迭代器一样（next() 会操心如何调用生成器的 __next__() 方法，不用你操心）。既然生成器是一个迭代器，它可以被用在 for 循环中。

每当生成器被调用的时候，它会返回一个值给调用者。在生成器内部使用 yield 来完成这个动作（例如 yield 7）。为了记住 yield 到底干了什么，最简单的方法是把它当作专门给生成器函数用的特殊的 return（加上点小魔法）。

下面是一个简单的生成器函数：  

```python
>>> def simple_generator_function():
>>>    yield 1
>>>    yield 2
>>>    yield 3
```

这里有两个简单的方法来使用它：

```python
>>> for value in simple_generator_function():
>>>     print(value)
1
2
3
>>> our_generator = simple_generator_function()
>>> next(our_generator)
1
>>> next(our_generator)
2
>>> next(our_generator)
3
```

## **魔法？**

那么神奇的部分在哪里？

我很高兴你问了这个问题！当一个生成器函数调用 yield，生成器函数的“状态”会被冻结，所有的变量的值会被保留下来，下一行要执行的代码的位置也会被记录，直到再次调用 next()。一旦 next() 再次被调用，生成器函数会从它上次离开的地方开始。如果永远不调用 next()，yield 保存的状态就被无视了。

我们来重写 get_primes() 函数，这次我们把它写作一个生成器。注意我们不再需要 magical_infinite_range 函数了。使用一个简单的 while 循环，我们创造了自己的无穷串列。

```python
def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1
```

如果生成器函数调用了 return，或者执行到函数的末尾，会出现一个 StopIteration 异常。 这会通知 next() 的调用者这个生成器没有下一个值了（这就是普通迭代器的行为）。这也是这个 while 循环在我们的 get_primes() 函数出现的原因。如果没有这个 while，当我们第二次调用 next() 的时候，生成器函数会执行到函数末尾，触发 StopIteration 异常。一旦生成器的值用完了，再调用 next() 就会出现错误，所以你只能将每个生成器的使用一次。下面的代码是错误的：

```python
>>> our_generator = simple_generator_function()
>>> for value in our_generator:
>>>     print(value)
 
>>> # 我们的生成器没有下一个值了...
>>> print(next(our_generator))
Traceback (most recent call last):
  File "<ipython-input-13-7e48a609051a>", line 1, in <module>
    next(our_generator)
StopIteration
 
>>> # 然而，我们总可以再创建一个生成器
>>> # 只需再次调用生成器函数即可
 
>>> new_generator = simple_generator_function()
>>> print(next(new_generator)) # 工作正常
1
```

因此，这个 while 循环是用来确保生成器函数永远也不会执行到函数末尾的。只要调用 next() 这个生成器就会生成一个值。这是一个处理无穷序列的常见方法（这类生成器也是很常见的）。

## **执行流程**

让我们回到调用 get_primes 的地方：solve_number_10。

```python
def solve_number_10():
    # She *is* working on Project Euler #10, I knew it!
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 2000000:
            total += next_prime
        else:
            print(total)
            return
```

我们来看一下 solve_number_10 的 for 循环中对 get_primes 的调用，观察一下前几个元素是如何创建的有助于我们的理解。当 for 循环从 get_primes 请求第一个值时，我们进入 get_primes，这时与进入普通函数没有区别。

1. 进入第三行的 while 循环
2. 停在 if 条件判断（3 是素数）
3. 通过 yield 将 3 和执行控制权返回给 solve_number_10

接下来，回到 insolve_number_10：

1. 循环得到返回值 3
2. for 循环将其赋给 next_prime
3. total 加上 next_prime
4. for 循环从 get_primes 请求下一个值

这次，进入 get_primes 时并没有从开头执行，我们从第 5 行继续执行，也就是上次离开的地方。

```python
def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1 # <<<<<<<<<<
```

最关键的是，number 还保持我们上次调用 yield 时的值（例如 3）。记住，yield 会将值传给 next() 的调用方，同时还会保存生成器函数的“状态”。接下来，number 加到 4，回到 while 循环的开始处，然后继续增加直到得到下一个素数（5）。我们再一次把 number 的值通过 yield 返回给 solve_number_10 的 for 循环。这个周期会一直执行，直到 for 循环结束（得到的素数大于2,000,000）。

## **更给力点**

在 PEP 342 中加入了将值传给生成器的支持。PEP 342 加入了新的特性，能让生成器在单一语句中实现，生成一个值（像从前一样），接受一个值，或同时生成一个值并接受一个值。

我们用前面那个关于素数的函数来展示如何将一个值传给生成器。这一次，我们不再简单地生成比某个数大的素数，而是找出比某个数的等比级数大的最小素数（例如 10， 我们要生成比 10，100，1000，10000 ... 大的最小素数）。我们从 get_primes 开始：  

```python
def print_successive_primes(iterations, base=10):
    # 像普通函数一样，生成器函数可以接受一个参数
    
    prime_generator = get_primes(base)
    # 这里以后要加上点什么
    for power in range(iterations):
        # 这里以后要加上点什么
 
def get_primes(number):
    while True:
        if is_prime(number):
        # 这里怎么写?
```

get_primes 的后几行需要着重解释。yield 关键字返回 number 的值，而像 other = yield foo 这样的语句的意思是，“返回 foo 的值，这个值返回给调用者的同时，将 other 的值也设置为那个值”。你可以通过 send 方法来将一个值“发送”给生成器。

```python
def get_primes(number):
    while True:
        if is_prime(number):
            number = yield number
        number += 1
```

通过这种方式，我们可以在每次执行 yield 的时候为 number 设置不同的值。现在我们可以补齐 print_successive_primes 中缺少的那部分代码：

```python
def print_successive_primes(iterations, base=10):
    prime_generator = get_primes(base)
    prime_generator.send(None)
    for power in range(iterations):
        print(prime_generator.send(base ** power))
```

  这里有两点需要注意：首先，我们打印的是 generator.send 的结果，这是没问题的，因为 send 在发送数据给生成器的同时还返回生成器通过 yield 生成的值（就如同生成器中 yield 语句做的那样）。

第二点，看一下 prime_generator.send(None) 这一行，当你用 send 来“启动”一个生成器时（就是从生成器函数的第一行代码执行到第一个 yield 语句的位置），你必须发送 None。这不难理解，根据刚才的描述，生成器还没有走到第一个 yield 语句，如果我们发生一个真实的值，这时是没有人去“接收”它的。一旦生成器启动了，我们就可以像上面那样发送数据了。


## **综述**

在本系列文章的后半部分，我们将讨论一些 yield 的高级用法及其效果。yield 已经成为 Python 最强大的关键字之一。现在我们已经对 yield 是如何工作的有了充分的理解，我们已经有了必要的知识，可以去了解 yield 的一些更“费解”的应用场景。

不管你信不信，我们其实只是揭开了 yield 强大能力的一角。例如，send 确实如前面说的那样工作，但是在像我们的例子这样，只是生成简单的序列的场景下，send 几乎从来不会被用到。下面我贴一段代码，展示 send 通常的使用方式。对于这段代码如何工作以及为何可以这样工作，在此我并不打算多说，它将作为第二部分很不错的热身。


```python
import random
 
def get_data():
    """返回0到9之间的3个随机数"""
    return random.sample(range(10), 3)
 
def consume():
    """显示每次传入的整数列表的动态平均值"""
    running_sum = 0
    data_items_seen = 0
 
    while True:
        data = yield
        data_items_seen += len(data)
        running_sum += sum(data)
        print('The running average is {}'.format(running_sum / float(data_items_seen)))
 
def produce(consumer):
    """产生序列集合，传递给消费函数（consumer）"""
    while True:
        data = get_data()
        print('Produced {}'.format(data))
        consumer.send(data)
        yield
 
if __name__ == '__main__':
    consumer = consume()
    consumer.send(None)
    producer = produce(consumer)
 
    for _ in range(10):
        print('Producing...')
        next(producer)
```

## **请谨记……**
我希望您可以从本文的讨论中获得一些关键的思想：

- generator 是用来产生一系列值的
- yield 则像是 generator 函数的返回结果
- yield 唯一所做的另一件事就是保存一个 generator 函数的状态
- generator 就是一个特殊类型的迭代器（iterator）
- 和迭代器相似，我们可以通过使用 next() 来从 generator 中获取下一个值
- 通过隐式地调用 next() 来忽略一些值

我希望这篇文章是有益的。如果您还从来没有听说过 generator，我希望现在您可以理解它是什么以及它为什么是有用的，并且理解如何使用它。如果您已经在某种程度上比较熟悉 generator，我希望这篇文章现在可以让您扫清对 generator 的一些困惑。

同往常一样，如果某一节的内容不是很明确（或者某节内容更重要，亦或某些内容包含错误），请尽一切办法让我知晓。您可以在下面留下您的评论、给发送电子邮件或在 Twitter中@jeffknupp。