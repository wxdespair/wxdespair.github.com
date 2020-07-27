# 1  C语言的简介

C语言是一种高效的、通用的、面向过程的计算机高级程序设计语言。C语言对操作系统和系统使用程序以及需要对硬件进行操作的场合，用C语言明显优于其它高级语言，许多大型应用软件都是用C语言编写的。1972年，为了移植与开发UNIX操作系统，丹尼斯·里奇在贝尔电话实验室设计开发了C语言。 

## 1.0  C语言发展史

在20世纪60年代

C语言诞生于美国的[贝尔](https://baike.baidu.com/item/贝尔/1064241)实验室，由D.M.Ritchie以[B语言](https://baike.baidu.com/item/B语言/1845842)为基础发展而来，在它的主体设计完成后，Thompson和Ritchie用它完全重写了UNIX，且随着UNIX的发展，c语言也得到了不断的完善。为了利于C语言的全面推广，许多专家学者和硬件厂商联合组成了C语言标准委员会，并在之后的1989年，诞生了第一个完备的C标准，简称“C89”，也就是“ANSI c”，截至2020年，最新的C语言标准为2017年发布的 “C17”。 

C语言之所以命名为C，是因为 C语言源自[Ken Thompson](https://baike.baidu.com/item/Ken Thompson)发明的[B语言](https://baike.baidu.com/item/B语言)，而 B语言则源自BCPL语言。

1967年，[剑桥大学](https://baike.baidu.com/item/剑桥大学)的Martin Richards对CPL语言进行了简化，于是产生了BCPL（Basic Combined Programming Language）语言。

20世纪60年代，美国[AT&T](https://baike.baidu.com/item/AT%26T)公司[贝尔实验室](https://baike.baidu.com/item/贝尔实验室)（AT&T Bell Laboratory）的研究员[Ken Thompson](https://baike.baidu.com/item/Ken Thompson)闲来无事，手痒难耐，想玩一个他自己编的，模拟在太阳系航行的电子游戏——Space Travel。他背着老板，找到了台空闲的机器——[PDP-7](https://baike.baidu.com/item/PDP-7)。但这台机器没有操作系统，而游戏必须使用操作系统的一些功能，于是他着手为PDP-7开发操作系统。后来，这个操作系统被命名为——[UNIX](https://baike.baidu.com/item/UNIX)。

1970年，美国贝尔实验室的 Ken Thompson，以[BCPL](https://baike.baidu.com/item/BCPL)语言为基础，设计出很简单且很接近硬件的B语言（取BCPL的首字母）。并且他用B语言写了第一个UNIX操作系统。

1971年，同样[酷爱](https://baike.baidu.com/item/酷爱/1371)Space Travel的[Dennis M.Ritchie](https://baike.baidu.com/item/Dennis M.Ritchie)为了能早点儿玩上游戏，加入了Thompson的开发项目，合作开发UNIX。他的主要工作是改造B语言，使其更成熟。

1972年，美国[贝尔实验室](https://baike.baidu.com/item/贝尔实验室)的 D.M.Ritchie 在B语言的基础上最终设计出了一种新的[语言](https://baike.baidu.com/item/语言/2291095)，他取了BCPL的第二个字母作为这种语言的名字，这就是C语言。

1973年初，C语言的主体完成。Thompson和Ritchie迫不及待地开始用它完全重写了[UNIX](https://baike.baidu.com/item/UNIX)。此时，编程的乐趣使他们已经完全忘记了那个"Space Travel"，一门心思地投入到了UNIX和C语言的开发中。随着UNIX的发展，C语言自身也在不断地完善。直到2020年，各种版本的UNIX内核和周边工具仍然使用C语言作为最主要的开发语言，其中还有不少继承Thompson和Ritchie之手的代码。 

在开发中，他们还考虑把[UNIX](https://baike.baidu.com/item/UNIX)移植到其他类型的计算机上使用。C语言强大的移植性（Portability）在此显现。机器语言和[汇编语言](https://baike.baidu.com/item/汇编语言/61826)都不具有移植性，为[x86](https://baike.baidu.com/item/x86)开发的程序，不可能在Alpha、[SPARC](https://baike.baidu.com/item/SPARC)和[ARM](https://baike.baidu.com/item/ARM/7518299)等机器上运行。而C语言程序则可以使用在任意架构的[处理器](https://baike.baidu.com/item/处理器)上，只要那种架构的处理器具有对应的C语言[编译器](https://baike.baidu.com/item/编译器)和库，然后将C源代码[编译](https://baike.baidu.com/item/编译)、[连接](https://baike.baidu.com/item/连接/8248019)成目标[二进制文件](https://baike.baidu.com/item/二进制文件)之后即可运行。 

1977年，[Dennis M.Ritchie](https://baike.baidu.com/item/Dennis M.Ritchie)发表了不依赖于具体机器系统的C语言编译文本《可移植的C语言编译程序》。

## 1.1  C语言标准

当前最新的C语言标准为C11，在它之前的C语言标准为C99。

1982年，很多有识之士和[美国国家标准协会](https://baike.baidu.com/item/美国国家标准协会)为了使这个语言健康地发展下去，决定成立C标准委员会，建立C语言的标准。委员会由硬件厂商、编译器及其他软件工具生产商、软件设计师、顾问、学术界人士、C语言作者和应用程序员组成。1989年，[ANSI](https://baike.baidu.com/item/ANSI/14955)发布了第一个完整的C语言标准——ANSI X3.159—1989，简称“C89”，不过人们也习惯称其为“[ANSI C](https://baike.baidu.com/item/ANSI C)”。C89在1990年被[国际标准组织](https://baike.baidu.com/item/国际标准组织)ISO(International Standard Organization)一字不改地采纳，ISO官方给予的名称为：ISO/IEC 9899，所以ISO/IEC9899: 1990也通常被简称为“C90”。1999年，在做了一些必要的修正和完善后，ISO发布了新的C语言标准，命名为ISO/IEC 9899：1999，简称“[C99](https://baike.baidu.com/item/C99)”。 [6] 在2011年12月8日，ISO又正式发布了新的标准，称为ISO/IEC9899: 2011，简称为“[C11](https://baike.baidu.com/item/C11)”。

## 1.2  C语言的主要特点

### 1.2.1优点

C语言是一种结构化语言，它有着清晰的层次，可按照模块的方式对程序进行编写，十分有利于程序的调试，且c语言的处理和表现能力都非常的强大，依靠非常全面的运算符和多样的[数据类型](https://baike.baidu.com/item/数据类型/10997964)，可以轻易完成各种数据结构的构建，通过指针类型更可对内存直接寻址以及对硬件进行直接操作，因此既能够用于开发系统程序，也可用于开发应用软件。

（1）语言使用简便

C语言包含的各种[控制语句](https://baike.baidu.com/item/控制语句/10507605)仅有9种，关键字也只有32 个，程序的编写要求不严格且以小写字母为主，对许多不必要的部分进行了精简。实际上，语句构成与硬件有关联的较少，且C语言本身不提供与硬件相关的输入输出、文件管理等功能，如需此类功能，需要通过配合编译系统所支持的各类库进行编程，故c语言拥有非常简洁的编译系统。 但相同的程序较其他语言也会拥有更多的语句。

（2）丰富的数据类型

C语言包含的数据类型广泛，不仅包含有传统的字符型、整型、浮点型、数组类型等数据类型，还具有其他编程语言所不具备的数据类型，其中以指针类型数据使用最为灵活，可以通过编程对各种数据结构进行计算。 

（3）可对[物理地址](https://baike.baidu.com/item/物理地址/2901583)进行直接操作

C语言允许对[硬件](https://baike.baidu.com/item/硬件/479446)内存地址进行直接读写，以此可以实现汇编语言的主要功能，并可直接操作硬件。C语言不但具备高级语言所具有的良好特性，又包含了许多低级语言的优势，故在系统软件编程领域有着广泛的应用。 

（4）代码具有较好的可移植性

C语言是面向过程的编程语言，用户只需要关注所被解决问题的本身，而不需要花费过多的精力去了解相关硬件，且针对不同的硬件环境，在用C语言实现相同功能时的代码基本一致，不需或仅需进行少量改动便可完成移植，这就意味着，对于一台计算机编写的C程序可以在另一台计算机上轻松地运行，从而极大的减少了程序移植的工作强度。 

（5）可生成高质量、目标代码执行效率高的程序

与其他高级语言相比，C语言可以生成高质量和高效率的目标代码，故通常应用于对[代码质量](https://baike.baidu.com/item/代码质量/8863758)和执行效率要求较高的[嵌入式系统](https://baike.baidu.com/item/嵌入式系统/186978)程序的编写。 

 （6）C语言是普适性最强的一种计算机程序编辑语言，它不仅可以发挥出高级编程语言的功用，还具有汇编语言的优点，因此相对于其它编程语言，它具有自己独特的特点。具体体现在以下三个方面：

其一，广泛性。C 语言的运算范围的大小直接决定了其优劣性。C 语言中包含了34种运算符，因此运算范围要超出许多其它语言，此外其运算结果的表达形式也十分丰富。此外，C 语言包含了字符型、[指针](https://baike.baidu.com/item/指针/2878304)型等多种数据结构形式，因此，更为庞大的数据结构运算它也可以应付。 

其二，简洁性。9 类控制语句和32个[KEYWORDS](https://baike.baidu.com/item/KEYWORDS/8284218)是C语言所具有的基础特性，使得其在计算机应用程序编写中具有广泛的适用性，不仅可以适用广大编程人员的操作，提高其工作效率，同 时还能够支持高级编程，避免了语言切换的繁琐。 

其三，结构完善。C语言是一种结构化语言，它可以通过组建模块单位的形式实现[模块化](https://baike.baidu.com/item/模块化/3295536)的应用程序，在系统描述方面具有显著优势，同时这一特性也使得它能够适应多种不同的编程要求，且执行效率高。 

### 1.2.2缺点

1. C语言的缺点主要表现在数据的封装性上，这一点使得C在数据的安全性上有很大缺陷，这也是C和C++的一大区别。

2. C语言的语法限制不太严格，对变量的类型约束不严格，影响程序的安全性，对[数组下标越界](https://baike.baidu.com/item/数组下标越界)不作检查等。从应用的角度，C语言比其他高级语言较难掌握。也就是说，对用C语言的人，要求对程序设计更熟练一些。 

